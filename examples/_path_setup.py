"""
Add the project root to sys.path so that 'from src...' works.
Use in notebooks: import _path_setup  (when kernel cwd is examples/)
Or copy the inline snippet from README into the first cell.
"""
import os
import sys


def find_repo_root():
    """Find repo root by looking for 'src' dir and 'requirements.txt'."""
    path = os.path.abspath(os.getcwd())
    for _ in range(10):
        if os.path.isdir(os.path.join(path, "src")) and os.path.exists(
            os.path.join(path, "requirements.txt")
        ):
            return path
        parent = os.path.dirname(path)
        if parent == path:
            break
        path = parent
    return os.path.abspath(os.path.join(os.getcwd(), ".."))


def setup_path():
    """Insert repo root at the start of sys.path. Idempotent."""
    repo_root = find_repo_root()
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    return repo_root


setup_path()
