import os.path

def check_path(save_path):
    """Ensures that a given path is valid, i.e. a string. Raises an error otherwise."""
    if not isinstance(save_path, str):
        raise TypeError(f"Expected 'savefig' to be string (existing file path), but got {type(save_path).__name__}.")

    save_dir = os.path.dirname(save_path)
    if save_dir and not os.path.exists(save_dir):
        raise FileNotFoundError(f"Path '{save_path}' does not exist.\n"
                                f"Please create it before saving, or give a valid path.")