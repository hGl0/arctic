import logging
import pytest

from vortexclust.io.paths import *

logger = logging.getLogger(__name__)

def test_check_path(tmp_path):
    file_path = tmp_path / "path"
    check_path(str(file_path))


def test_check_path_invalid(tmp_path):
    with pytest.raises(TypeError, match="Expected a string"):
        check_path(12345)

    nonexistent_dir = tmp_path / "does_not_exist"
    file_path = nonexistent_dir / "file.csv"

    with pytest.raises(FileNotFoundError, match="Path"):
        check_path(str(file_path))