import os.path as osp

import rmvd.utils as utils
from rmvd.utils import get_path


def get_paths_file():
    local_paths_file = osp.join(osp.dirname(osp.realpath(__file__)), "paths.toml")
    home_paths_file = osp.join(osp.expanduser('~'), 'rmvd_model_paths.toml')
    return utils.get_paths_file(local_paths_file, home_paths_file, "rmvd/models/README.md")


def get_repo_path(model_name):
    """Get the local repository path of a wrapped model from the paths.toml file."""
    paths_file = get_paths_file()
    repo_path = get_path(paths_file, model_name, "root")

    if repo_path is None:
        raise ValueError(f"No root path for the model {model_name} in the paths file {paths_file}. Please add a "
                         f"[{model_name}] section with a root entry, as specified in the rmvd/models/README.md file.")

    if not osp.isdir(repo_path):
        raise NotADirectoryError(f"The root path {repo_path} of the model {model_name} from the paths file "
                                 f"{paths_file} is not an existing directory. Please check the paths file and the "
                                 f"setup instructions in the rmvd/models/README.md file.")

    return repo_path
