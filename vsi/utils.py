import os
import sys
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


def file_compare(dir1, dir2):
    """
    To compare files in two different directory,
    notice that files in dir1 is the benchmark
    and files in dir2 is compared to dir2.
    Also, this function only compare the files' name
    not including the suffix of the file
    """
    bench_list = os.listdir(dir1)
    compare_list = os.listdir(dir2)
    print(sorted(bench_list))
    print(sorted(compare_list))
    print(len(bench_list))
    print(len(compare_list))

    bench_name_list = [i.split('.')[0] for i in bench_list]
    print(bench_name_list)
    exclude_list = [i.split('.')[0] for i in compare_list if i.split('.')[0] not in bench_name_list]
    print(sorted(exclude_list))
    print(len(exclude_list))


def files_count(dir):

    count = 0
    for root,subdirs,files in os.walk(dir):
        count += len(files)

    print(count)


def suffix_change(root_dir, old, new):
    """
    To change the suffix of files under given root
    :param root_dir: the directory of files
    :param old:
    :param new: 
    :return:
    Usage:
        >>> all_vsi_files = get_files(".", "vsi","tif")
        This will change all the .vsi files in the current
        directory to '.tif' files.
    """

    # sys.path.append(root_dir)
    # print(sys.path)
    os.chdir(root_dir)
    # 列出当前目录下所有的文件
    files = os.listdir(root_dir)  # 如果path为None，则使用path = '.'
    # print(files)

    for filename in files:
        print(1)
        sfx = filename.split('.')[-1]  # 分离文件名与扩展名
        # 如果后缀是jpg
        if sfx is old:
            # 重新组合文件名和后缀名
            newname = filename.split('.')[0] + '.' + new
            os.rename(filename,
                      newname)




if __name__ == '__main__':
    # root_path = get_config('config.yaml')
    #
    # file_paths = get_files(directory=root_path, keyword='.vsi')
    # a_file = file_paths[0]
    # print(a_file)

    # file_compare(dir1='/data/images/pathology/temp/Tianjinzhongliu/tif/',
    #              dir2='/data/images/pathology/temp/TjAnnotation/')
    # files_count('/data/temp/yangpengshuai/PIR_KIMIA_data/test/')

    suffix_change(root_dir='/data/images/pathology/temp/TjTiff/',
                  old='xml',
                  new='tif')