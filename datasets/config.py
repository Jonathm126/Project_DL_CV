import os

# Get the absolute path to the directory where config.py is located
config_dir = os.path.dirname(os.path.abspath(__file__))

# Assume that the project root is the parent directory of the 'datasets' folder
project_root = os.path.dirname(config_dir)

# # Now create dataset_path relative to project root
dataset_path = os.path.join(project_root, "datasets")
#dataset_path = 'C:\\Users\\jonathan\\.pytorch-datasets'

config = {
    "dataset_path": dataset_path
}