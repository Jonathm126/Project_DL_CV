import os

config_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(config_dir)
dataset_path = os.path.join(project_root, "datasets")

config = {
    "dataset_path": dataset_path
}
