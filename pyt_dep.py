import yaml
import subprocess
import sys
import os

def install_packages_from_yaml(yaml_file, venv_path="venv"):
    """Installs ONLY pip packages from environment.yml inside a virtual environment."""
    try:
        # Ensure the YAML file exists
        with open(yaml_file, 'r') as f:
            env_data = yaml.safe_load(f)

        dependencies = env_data.get('dependencies', [])
        pip_dependencies = []

        for dep in dependencies:
            if isinstance(dep, dict) and 'pip' in dep:
                pip_dependencies.extend(dep['pip'])  # Extract pip dependencies

        if pip_dependencies:
            print("Installing pip packages in virtual environment:")
            for package in pip_dependencies:
                print(f"  - {package}")

            # Construct the path to venv's pip
            pip_path = os.path.join(venv_path, "Scripts", "pip") if os.name == "nt" else os.path.join(venv_path, "bin", "pip")

            # Install dependencies using venv's pip
            subprocess.check_call([pip_path, "install"] + pip_dependencies)
        else:
            print("No pip dependencies found in the environment.yml file.")

    except FileNotFoundError:
        print(f"Error: The file {yaml_file} was not found.")
    except yaml.YAMLError as e:
        print(f"Error: Could not parse the YAML file. {e}")
    except subprocess.CalledProcessError as e:
        print(f"Error: Failed to install packages. {e}")

# Call function (Ensure you are in the correct directory)
install_packages_from_yaml('environment.yml')
