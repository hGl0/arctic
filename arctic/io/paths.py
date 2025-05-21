import os
def check_path(save_path: str) -> None:
    r"""
    Ensures that a given path is valid and exists. Raises an error otherwise.

    :param save_path: The file path where data will be saved.

    :raises TypeError: If 'save_path' is not a string.
    :raises FileNotFoundError: If the directory does not exist.

    :return: None
    """
    if not isinstance(save_path, str):
        raise TypeError(f"Expected a string (existing file path), but got {type(save_path).__name__}.")

    save_dir = os.path.dirname(save_path)
    if save_dir and not os.path.exists(save_dir):
        raise FileNotFoundError(f"Path '{save_path}' does not exist.\n"
                                f"Please create it before saving, or give a valid path.")
