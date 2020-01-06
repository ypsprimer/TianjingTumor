import os
from ruamel import yaml
import javabridge as jb


def get_config(file_name):
    """
    return configurations from config.yaml
    Usage:
        >>> root_path = get_config('config.yaml')
        This will get the root_path dumped in the yaml file
    """

    with open(file_name,'r') as file:
        content = yaml.load(file.read(), Loader=yaml.Loader)

    return content['file_path']


def get_files(directory, keyword):
    """ Returns all the files in the given directory
    and subdirectories, filtering with the keyword.
    Usage:
        >>> all_vsi_files = get_files(".", ".vsi")
        This will have all the .vsi files in the current
        directory and all other directories in the current
        directory.
    """

    file_list = []
    for path, subdirs, files in os.walk(directory):
        for name in files:
            filename = os.path.join(path, name)
            if keyword in filename:
                file_list.append(filename)
    return sorted(file_list)


if __name__ == '__main__':
    root_path = get_config('config.yaml')

    file_paths = get_files(directory=root_path, keyword='.vsi')
    a_file = file_paths[0]
    print(a_file)