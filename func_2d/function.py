# func_2d/function.py
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import cfg
from conf import settings
from func_2d.utils import *
import pandas as pd


args = cfg.parse_args()

device = torch.device("cpu")
GPUdevice = torch.device("cpu")
criterion_G = torch.nn.CrossEntropyLoss()
mask_type = torch.float32
torch.backends.cudnn.benchmark = True


def train_sam(args, net: nn.Module, optimizer, train_loader, epoch, writer):
    # torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
    # if torch.cuda.get_device_properties(0).major >= 8:
    #     torch.backends.cuda.matmul.allow_tf32 = True
    #     torch.backends.cudnn.allow_tf32 = True

    net.train()
    optimizer.zero_grad()
    epoch_loss = 0
    memory_bank_list = []
    lossfunc = criterion_G
    feat_sizes = [(256, 256), (128, 128), (64, 64)]

    with tqdm(total=len(train_loader), desc=f'Epoch {epoch}', unit='img') as pbar:
        for ind, pack in enumerate(train_loader):
            to_cat_memory = []
            to_cat_memory_pos = []
            to_cat_image_embed = []

            imgs = pack['image'].to(dtype=mask_type, device=GPUdevice)
            masks = pack['mask'].to(dtype=torch.long, device=GPUdevice)
            name = pack['image_meta_dict']['filename_or_obj']

            point_label, pt = None, None
            if 'pt' in pack:
                pt = pack['pt'].to(device=GPUdevice).unsqueeze(1)
                point_labels = pack['p_label'].to(device=GPUdevice).unsqueeze(1)
                coords_torch = torch.as_tensor(pt, dtype=torch.float, device=GPUdevice)
                labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=GPUdevice)
            else:
                coords_torch, labels_torch = None, None

            backbone_out = net.forward_image(imgs)
            _, vision_feats, vision_pos_embeds, _ = net._prepare_backbone_features(backbone_out)
            B = vision_feats[-1].size(1)

            if not memory_bank_list:
                vision_feats[-1] += torch.nn.Parameter(torch.zeros(1, B, net.hidden_dim)).to(device=GPUdevice)
                vision_pos_embeds[-1] += torch.nn.Parameter(torch.zeros(1, B, net.hidden_dim)).to(device=GPUdevice)
            else:
                memory_stack_ori = torch.stack([elem[0].to(GPUdevice, non_blocking=True).flatten(2).permute(2, 0, 1) for elem in memory_bank_list], dim=0)
                memory_pos_stack_ori = torch.stack([elem[1].to(GPUdevice, non_blocking=True).flatten(2).permute(2, 0, 1) for elem in memory_bank_list], dim=0)
                image_embed_stack_ori = torch.stack([elem[3].to(GPUdevice, non_blocking=True) for elem in memory_bank_list], dim=0)

                vision_feats_temp = vision_feats[-1].permute(1, 0, 2).reshape(B, -1, 64, 64).reshape(B, -1)
                image_embed_stack_ori = F.normalize(image_embed_stack_ori, p=2, dim=1)
                vision_feats_temp = F.normalize(vision_feats_temp, p=2, dim=1)
                similarity_scores = torch.mm(image_embed_stack_ori, vision_feats_temp.t()).t()
                similarity_scores = F.softmax(similarity_scores, dim=1)
                sampled_indices = torch.multinomial(similarity_scores, num_samples=B, replacement=True).squeeze(1)

                memory = memory_stack_ori[sampled_indices].squeeze(3).permute(1, 2, 0, 3).reshape(-1, memory_stack_ori.size(2), memory_stack_ori.size(3))
                memory_pos = memory_pos_stack_ori[sampled_indices].squeeze(3).permute(1, 2, 0, 3).reshape(-1, memory_stack_ori.size(2), memory_stack_ori.size(3))

                vision_feats[-1] = net.memory_attention(
                    curr=[vision_feats[-1]],
                    curr_pos=[vision_pos_embeds[-1]],
                    memory=memory,
                    memory_pos=memory_pos,
                    num_obj_ptr_tokens=0
                )

            feats = [feat.permute(1, 2, 0).view(B, -1, *feat_size) for feat, feat_size in zip(vision_feats[::-1], feat_sizes[::-1])][::-1]
            image_embed = feats[-1]
            high_res_feats = feats[:-1]

            with torch.no_grad():
                points = (coords_torch, labels_torch) if (ind % 5) == 0 else None
                se, de = net.sam_prompt_encoder(points=points, boxes=None, masks=None, batch_size=B)

            masks_from_decoder, iou_predictions, sam_output_tokens, object_score_logits = net.sam_mask_decoder(
                image_embeddings=image_embed,
                image_pe=net.sam_prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=se,
                dense_prompt_embeddings=de,
                multimask_output=True,
                repeat_image=False,
                high_res_features=high_res_feats
            )
            print(f"DEBUG (func_2d/function.py): masks_from_decoder.shape = {masks_from_decoder.shape}") # **ADD THIS DEBUG PRINT**


            # Prediction - use 'masks_from_decoder' as 'pred' (25-channel output)
            pred = F.interpolate(masks_from_decoder, size=(args.out_size, args.out_size))
            high_res_multimasks = F.interpolate(masks_from_decoder, size=(args.image_size, args.image_size), mode="bilinear", align_corners=False)

            maskmem_features, maskmem_pos_enc = net._encode_new_memory(
                current_vision_feats=vision_feats,
                feat_sizes=feat_sizes,
                pred_masks_high_res=pred,
                is_mask_from_pts=(ind % 5) == 0
            )

            maskmem_features = maskmem_features.to(torch.bfloat16).to(device=GPUdevice, non_blocking=True)
            maskmem_pos_enc = maskmem_pos_enc[0].to(torch.bfloat16).to(device=GPUdevice, non_blocking=True)

            if len(memory_bank_list) < args.memory_bank_size:
                for batch in range(maskmem_features.size(0)):
                    memory_bank_list.append([maskmem_features[batch].unsqueeze(0).detach(), maskmem_pos_enc[batch].unsqueeze(0).detach(), iou_predictions[batch, 0], image_embed[batch].reshape(-1).detach()])
            else:
                for batch in range(maskmem_features.size(0)):
                    memory_bank_maskmem_features_flatten = torch.stack([element[0].reshape(-1) for element in memory_bank_list])
                    memory_bank_maskmem_features_norm = F.normalize(memory_bank_maskmem_features_flatten, p=2, dim=1)
                    current_similarity_matrix = torch.mm(memory_bank_maskmem_features_norm, memory_bank_maskmem_features_norm.t())
                    current_similarity_matrix_no_diag = current_similarity_matrix.clone()
                    diag_indices = torch.arange(current_similarity_matrix_no_diag.size(0))
                    current_similarity_matrix_no_diag[diag_indices, diag_indices] = float('-inf')
                    single_key_norm = F.normalize(maskmem_features[batch].reshape(-1), p=2, dim=0).unsqueeze(1)
                    similarity_scores = torch.mm(memory_bank_maskmem_features_norm, single_key_norm).squeeze()
                    min_similarity_index = torch.argmin(similarity_scores)
                    max_similarity_index = torch.argmax(current_similarity_matrix_no_diag[min_similarity_index])
                    if similarity_scores[min_similarity_index] < current_similarity_matrix_no_diag[min_similarity_index][max_similarity_index]:
                        if iou_predictions[batch, 0] > memory_bank_list[max_similarity_index][2] - 0.1:
                            memory_bank_list.pop(max_similarity_index)
                            memory_bank_list.append([maskmem_features[batch].unsqueeze(0).detach(), maskmem_pos_enc[batch].unsqueeze(0).detach(), iou_predictions[batch, 0], image_embed[batch].reshape(-1).detach()])

            print(f"DEBUG: pred.shape = {pred.shape}, masks.shape = {masks.shape}")  # Debug print to check pred.shape
            loss = lossfunc(pred, masks.argmax(dim=1))
            pbar.set_postfix(**{'loss (batch)': loss.item()})
            epoch_loss += loss.item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            pbar.update()

    return epoch_loss / len(train_loader)


def validation_sam(args, val_loader, epoch, net: nn.Module, writer, clean_dir=True):
    # torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
    # if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
    #     torch.backends.cuda.matmul.allow_tf32 = True
    #     torch.backends.cudnn.allow_tf32 = True
    # else:
    #     print("CUDA not available, running on CPU.")

    net.eval()
    n_val = len(val_loader)
    threshold = (0.1, 0.3, 0.5, 0.7, 0.9)
    GPUdevice = torch.device('cpu')

    lossfunc = criterion_G
    memory_bank_list = []
    feat_sizes = [(256, 256), (128, 128), (64, 64)]
    total_loss, total_eiou, total_dice = 0, 0, 0

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for ind, pack in enumerate(val_loader):
            to_cat_memory = []
            to_cat_memory_pos = []
            to_cat_image_embed = []

            name = pack['image_meta_dict']['filename_or_obj']
            imgs = pack['image'].to(dtype=torch.float32, device=GPUdevice)
            masks = pack['mask'].to(dtype=torch.long, device=GPUdevice)

            point_label, pt = None, None
            if 'pt' in pack:
                pt = pack['pt'].to(device=GPUdevice).unsqueeze(1)
                point_labels = pack['p_label'].to(device=GPUdevice).unsqueeze(1)
                coords_torch = torch.as_tensor(pt, dtype=torch.float, device=GPUdevice)
                labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=GPUdevice)
            else:
                coords_torch, labels_torch = None, None

            with torch.no_grad():
                backbone_out = net.forward_image(imgs)
                _, vision_feats, vision_pos_embeds, _ = net._prepare_backbone_features(backbone_out)
                B = vision_feats[-1].size(1)

                if not memory_bank_list:
                    vision_feats[-1] += torch.nn.Parameter(torch.zeros(1, B, net.hidden_dim)).to(device=GPUdevice)
                    vision_pos_embeds[-1] += torch.nn.Parameter(torch.zeros(1, B, net.hidden_dim)).to(device=GPUdevice)
                else:
                    memory_stack_ori = torch.stack([elem[0].to(GPUdevice, non_blocking=True).flatten(2).permute(2, 0, 1) for elem in memory_bank_list], dim=0)
                    memory_pos_stack_ori = torch.stack([elem[1].to(GPUdevice, non_blocking=True).flatten(2).permute(2, 0, 1) for elem in memory_bank_list], dim=0)
                    image_embed_stack_ori = torch.stack([elem[3].to(GPUdevice, non_blocking=True) for elem in memory_bank_list], dim=0)

                    vision_feats_temp = vision_feats[-1].permute(1, 0, 2).view(B, -1, 64, 64).reshape(B, -1)
                    image_embed_stack_ori = F.normalize(image_embed_stack_ori, p=2, dim=1)
                    vision_feats_temp = F.normalize(vision_feats_temp, p=2, dim=1)
                    similarity_scores = torch.mm(image_embed_stack_ori, vision_feats_temp.t()).t()
                    similarity_scores = F.softmax(similarity_scores, dim=1)
                    sampled_indices = torch.multinomial(similarity_scores, num_samples=B, replacement=True).squeeze(1)

                    memory = memory_stack_ori[sampled_indices].squeeze(3).permute(1, 2, 0, 3).reshape(-1, memory_stack_ori.size(2), memory_stack_ori.size(3))
                    memory_pos = memory_pos_stack_ori[sampled_indices].squeeze(3).permute(1, 2, 0, 3).reshape(-1, memory_stack_ori.size(2), memory_stack_ori.size(3))

                    vision_feats[-1] = net.memory_attention(
                        curr=[vision_feats[-1]],
                        curr_pos=[vision_pos_embeds[-1]],
                        memory=memory,
                        memory_pos=memory_pos,
                        num_obj_ptr_tokens=0
                    )

                feats = [feat.permute(1, 2, 0).view(B, -1, *feat_size) for feat, feat_size in zip(vision_feats[::-1], feat_sizes[::-1])][::-1]
                image_embed = feats[-1]
                high_res_feats = feats[:-1]

                points = (coords_torch, labels_torch) if (ind % 5) == 0 else None
                se, de = net.sam_prompt_encoder(points=points, boxes=None, masks=None, batch_size=B)

                masks_from_decoder, iou_predictions, sam_output_tokens, object_score_logits = net.sam_mask_decoder(
                    image_embeddings=image_embed,
                    image_pe=net.sam_prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=se,
                    dense_prompt_embeddings=de,
                    multimask_output=True,
                    repeat_image=False,
                    high_res_features=high_res_feats
                )
                print(f"DEBUG (func_2d/function.py): masks_from_decoder.shape = {masks_from_decoder.shape}") # **ADD THIS DEBUG PRINT**


                # Prediction - use 'masks_from_decoder' as 'pred' (25-channel output)
                pred = F.interpolate(masks_from_decoder, size=(args.out_size, args.out_size))
                high_res_multimasks = F.interpolate(masks_from_decoder, size=(args.image_size, args.image_size), mode="bilinear", align_corners=False)

                maskmem_features, maskmem_pos_enc = net._encode_new_memory(
                    current_vision_feats=vision_feats,
                    feat_sizes=feat_sizes,
                    pred_masks_high_res=pred,
                    is_mask_from_pts=(ind % 5) == 0
                )
                    
                maskmem_features = maskmem_features.to(torch.bfloat16).to(device=GPUdevice, non_blocking=True)
                maskmem_pos_enc = maskmem_pos_enc[0].to(torch.bfloat16).to(device=GPUdevice, non_blocking=True)

                if len(memory_bank_list) < 16:
                    for batch in range(maskmem_features.size(0)):
                        memory_bank_list.append([maskmem_features[batch].unsqueeze(0), maskmem_pos_enc[batch].unsqueeze(0), iou_predictions[batch, 0], image_embed[batch].reshape(-1).detach()])
                else:
                    for batch in range(maskmem_features.size(0)):
                        memory_bank_maskmem_features_flatten = torch.stack([element[0].reshape(-1) for element in memory_bank_list])
                        memory_bank_maskmem_features_norm = F.normalize(memory_bank_maskmem_features_flatten, p=2, dim=1)
                        current_similarity_matrix = torch.mm(memory_bank_maskmem_features_norm, memory_bank_maskmem_features_norm.t())
                        current_similarity_matrix_no_diag = current_similarity_matrix.clone()
                        diag_indices = torch.arange(current_similarity_matrix_no_diag.size(0))
                        current_similarity_matrix_no_diag[diag_indices, diag_indices] = float('-inf')
                        single_key_norm = F.normalize(maskmem_features[batch].reshape(-1), p=2, dim=0).unsqueeze(1)
                        similarity_scores = torch.mm(memory_bank_maskmem_features_norm, single_key_norm).squeeze()
                        min_similarity_index = torch.argmin(similarity_scores)
                        max_similarity_index = torch.argmax(current_similarity_matrix_no_diag[min_similarity_index])
                        if similarity_scores[min_similarity_index] < current_similarity_matrix_no_diag[min_similarity_index][max_similarity_index]:
                            if iou_predictions[batch, 0] > memory_bank_list[max_similarity_index][2] - 0.1:
                                memory_bank_list.pop(max_similarity_index)
                                memory_bank_list.append([maskmem_features[batch].unsqueeze(0), maskmem_pos_enc[batch].unsqueeze(0), iou_predictions[batch, 0], image_embed[batch].reshape(-1).detach()])

                print(f"DEBUG: pred.shape = {pred.shape}, masks.shape = {masks.shape}") # Debug print - check pred.shape
                total_loss += lossfunc(pred, masks.argmax(dim=1))
                pred_binary = (pred > 0.5).float()
                # Unpack the evaluation metrics (assuming eval_seg returns a tuple of two scalars)
                (eiou_tuple, dice_tuple) = eval_seg(pred_binary, masks, threshold)
                # Compute the average over all 25 values:
                eiou_scalar = np.mean(eiou_tuple)
                dice_scalar = np.mean(dice_tuple)
                total_eiou += eiou_scalar
                total_dice += dice_scalar


                if args.vis > 0 and ind % args.vis == 0:
                    namecat = 'Test'
                    for na in name:
                        img_name = na
                        namecat = namecat + img_name + '+'
                    vis_image(imgs, pred_binary, masks, os.path.join(args.path_helper['sample_path'], namecat + 'epoch+' + str(epoch) + '.jpg'), reverse=False, points=None)

            pbar.update()

    return total_loss / n_val, tuple([total_eiou / n_val, total_dice / n_val])