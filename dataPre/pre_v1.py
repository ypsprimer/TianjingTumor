import numpy as np
import os
import shutil
import random

import cv2
import matplotlib.pyplot as plt

from xml.etree.ElementTree import parse
import openslide

from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# slide_path = '/data/images/pathology/temp/TjTiff/155-477366-1.tif'
# slide_path = '/data/images/pathology/temp/TjTiff/505-538588-5.tif'
slide_path = '/data/images/pathology/temp/TjTiff/813-539095-6.tif'
# note_path = '/data/images/pathology/temp/TjAnnotation/155-477366-1.xml'
# note_path = '/data/images/pathology/temp/TjAnnotation/505-538588-5.xml'
note_path = '/data/images/pathology/temp/TjAnnotation/813-539095-6.xml'

mask_path = '/temp/zhaiyupeng/tianjing/v1/mask/'
map_path = '/temp/zhaiyupeng/tianjing/v1/map/'
patch_path = '/temp/zhaiyupeng/tianjing/v1/patch_l3/'

single_file = '813-539095-6.png'

ROOT_DIR = '/data/images/pathology/temp/TjTiff/'
ANOTE_DIR = '/data/images/pathology/temp/TjAnnotation/'

MAP_DIR = '/temp/zhaiyupeng/tianjing/v1/map/'
MASK_DIR = '/temp/zhaiyupeng/tianjing/v1/mask/'

PATCH_DIR = '/temp/zhaiyupeng/tianjing/v1/'




def read_xml(file_path, mask_level):
    """
    To read coordinates in xml files
    :param file_path:
    :param mask_level:
    :return: a list
    """

    xml = parse(file_path).getroot()
    coors_list = []
    coors = []
    for areas in xml.iter('Coordinates'):
        for area in areas:
            coors.append([round(float(area.get('X')) / (2 ** mask_level)),
                          round(float(area.get('Y')) / (2 ** mask_level))])
        coors_list.append(coors)
        coors = []
    return coors_list


def make_mask(slide_path, note_path ,mask_level, mask_path, map_path):
    """

    :param slide_path:
    :param note_path:
    :param mask_level:
    :param mask_path:
    :param map_path:
    :return:
    """
    # name of the file excluding the suffix
    file_name = slide_path.split('/')[-1].split('.')[0]
    # slide loading
    slide = openslide.OpenSlide(slide_path)
    slide_thumbnail = np.array(slide.get_thumbnail(slide.level_dimensions[mask_level]))
    # print(type(slide_thumbnail))

    # xml loading
    coors_list = read_xml(note_path, mask_level)

    # draw boundary of annotation in the thumbnail
    # and return an RGB image
    for coors in coors_list:
        # print(np.array([coors]))
        # -1: 绘制所有  5: 轮廓粗细
        cv2.drawContours(slide_thumbnail, np.array([coors]), -1, (0,255,0), 10)
    # save thumbnail with the annotaion
    cv2.imwrite(map_path, slide_thumbnail)


    # print(slide.level_dimensions[mask_level])
    # print(slide.level_dimensions[mask_level][::-1])

    # draw the area covered by annotation and return a black-and-white image
    region_mask = np.zeros(slide.level_dimensions[mask_level][::-1]) # To transpose width and height
    for coors in coors_list:
        cv2.drawContours(region_mask, np.array([coors]), -1, 255, -1)
    cv2.imwrite(mask_path, region_mask)

    return slide_thumbnail, region_mask


def make_patch(slide_path, mask_path, map_path, patch_root, patch_size, raw_level, mask_level):
    """
    To generate patch using mask and threshold
    :param slide_path:
    :param mask_path:
    :param patch_root:
    :param patch_size:
    :param mask_level:
    :return:
    """
    if not os.path.exists(patch_root):
        os.makedirs(patch_root)

    slide = openslide.OpenSlide(slide_path)
    file_name = slide_path.split('/')[-1].split('.')[0]
    # print(slide.level_downsamples)
    # print(slide.level_dimensions)
    region_mask = cv2.imread(mask_path, 0)
    slide_map = cv2.imread(map_path)

    print('Size of slide:{}'.format(slide.level_dimensions[raw_level]))

    width, height = np.array(slide.level_dimensions[raw_level]) // patch_size
    print('Patches on width:{}, Patches on height:{}'.format(width, height))
    total = width * height
    all_cnt = 0
    t_cnt = 0
    # n_cnt = 0
    t_over = False
    # n_over = False

    # step represents the patch_size in the slide map
    step = int(patch_size / (2 ** (mask_level-raw_level)))
    # step = int(patch_size / 4)

    file_name = slide_path.split('/')[-1].split('.')[0]

    for i in range(width):
        for j in range(height):
            pixel_sum = region_mask[step * j: step * (j + 1),
                        step * i: step * (i + 1)].sum()
            pixel_max = step * step * 255
            nonzero_ratio = pixel_sum / pixel_max

            if (nonzero_ratio >= 0.9) and not t_over:
                t_cnt += 1
                # t_over = (t_cnt > tumor_sel_max)
                patch_name = os.path.join(patch_root, file_name  + '_' + str(i) + '_' + str(j) + '_.png')
                patch = slide.read_region((patch_size * i * (2**raw_level), patch_size * j * (2**raw_level)),
                                          raw_level,
                                          (patch_size, patch_size))
                # patch = np.array(patch)
                # print(type(patch))
                patch.save(patch_name)
                # cv2.imwrite(patch_name, np.array(patch))
                cv2.rectangle(slide_map, (step * i, step * j), (step * (i + 1), step * (j + 1)), (0, 0, 255), 5)

            all_cnt += 1
            print('\rProcess: %.3f%%,  All: %d, Extracted: %d'
                  % ((100. * all_cnt / total), all_cnt, t_cnt), end="")

    cv2.imwrite(map_path + file_name + '.png', slide_map)

    return t_cnt



def img_plot(image, image_name):
    """
    To plot a image
    :param image: numpy.ndarray
    :param title: the title of this plot
    :return:
    """
    name = image_name.split('/')[-1].split('.')[0]

    plt.figure()
    plt.title(name)
    plt.imshow(image)
    plt.show()


def region_plot(slide_path, coor, level, patch_size):

    slide = openslide.OpenSlide(slide_path)
    patch = slide.read_region((0, 11776), level, (patch_size, patch_size))
    # patch.save('/temp/zhaiyupeng/tianjing/v1/patch/2.png')
    img_plot(np.array(patch), slide_path)


def data_check(dir1, dir2, name):
    """
    To check the existence of files and annotations,
    if any of files or annotation do not exist, return False
    :param dir1:
    :param dir2:
    :param name:
    :return:
    """
    file_in_dir1 = os.path.join(dir1, name + '.tif')
    file_in_dir2 = os.path.join(dir2, name + '.xml')
    if os.path.exists(file_in_dir1) and os.path.exists(file_in_dir2):
        return True
    else:
        return False


def data_generate(root, note_root,
                  map_path='', mask_path='', patch_path='',
                  raw_level=2, mask_level=4,
                  p_size=256):
    """
    To generate data
    :param root:
    :param note_root:
    :param map_path:
    :param mask_path:
    :param patch_path:
    :param level:
    :param p_size:
    :return:
    """
    files = os.listdir(root)
    count_dict = {}
    # print(len(files))
    for file in tqdm(files, total=len(files)):
        file_name = file.split('/')[-1].split('.')[0]
        if not data_check(root,note_root,file_name):
            # print(file_name)
            continue
        else:
            thumbnail, mask = make_mask(slide_path=os.path.join(root,file_name + '.tif'),
                                        note_path=os.path.join(note_root, file_name + '.xml'),
                                        mask_level=mask_level,
                                        mask_path=os.path.join(mask_path, file_name + '.png'),
                                        map_path=os.path.join(map_path, file_name + '.png'),
                                        )

            count=make_patch(slide_path=os.path.join(root, file_name + '.tif'),
                             mask_path=os.path.join(mask_path, file_name + '.png'),
                             map_path=os.path.join(map_path, file_name + '.png'),
                             patch_root=os.path.join(patch_path, 'l{}'.format(raw_level), file_name),
                             patch_size=p_size,
                             raw_level=raw_level,
                             mask_level=mask_level,
                             )
            count_dict[file_name] = count
    print(count)


def break_restore(root_dir, obj_dir):

    root_files = os.listdir(root_dir)
    obj_files = os.listdir(obj_dir)
    name_list = [a.split('.')[0] for a in root_files]
    for name in name_list:
        if name not in obj_files:
            print(name)


def data_divide(root_dir, chosen, test_dir, train_dir, valid_dir):
    """
    To divide data into train, test and validate set
    :param root_dir:
    :param chosen:
    :param obj_dir:
    :param obj_split_dir:
    :return:
    """
    # To split the test set using the chosen files
    for name in tqdm(chosen, total=len(chosen)):
        source_path = os.path.join(root_dir, name)
        des_path = os.path.join(test_dir,name)
        if os.path.exists(source_path) and not os.path.exists(des_path):
            # print(source_path)
            shutil.copytree(source_path, des_path)
    print('Test set part1 successfully split!')

    # To split train and validate set
    all_list = os.listdir(root_dir)
    # print(len(all_list))
    for name in chosen:
        # To remove the test set
        all_list.remove(name)

    psp_list = [a for a in all_list if a[0]=='1' or a[0]=='2']
    other_list = [b for b in all_list if b[0]=='5' or b[0]=='8']
    random.seed(2020)
    psp_chosen_test = random.sample(psp_list, 20)
    # # To add a special slide into the train set
    # if '230-534771-5' not in psp_chosen:
    #     psp_chosen += ['230-534771-5']
    psp_chosen = psp_list.copy()
    for i in psp_chosen_test:
        psp_chosen.remove(i)
    print(psp_chosen)
    print(len(other_list))
    train_list = psp_chosen + other_list
    print(len(train_list))

    # To split the test set using the chosen files
    for name in tqdm(psp_chosen_test, total=len(psp_chosen_test)):
        source_path = os.path.join(root_dir, name)
        des_path = os.path.join(test_dir, name)
        if os.path.exists(source_path) and not os.path.exists(des_path):
            # print(source_path)
            shutil.copytree(source_path, des_path)
    print('Test set part2 successfully split!')

    for name in tqdm(train_list, total=len(train_list)):
        files = os.listdir(os.path.join(root_dir, name))
        split_files = random.sample(files,(len(files) // 10))
        for split_file in split_files:
            files.remove(split_file)
        # print(files)
        if not os.path.exists(os.path.join(train_dir, name)):
            os.makedirs(os.path.join(train_dir,name))
        if not os.path.exists(os.path.join(valid_dir, name)):
            os.makedirs(os.path.join(valid_dir, name))

        for a in files:
            if not os.path.exists(os.path.join(train_dir, name, a)):
                shutil.copy(os.path.join(root_dir,name,a),
                            os.path.join(train_dir,name,a))
        for a in split_files:
            if not os.path.exists(os.path.join(valid_dir, name, a)):
                shutil.copy(os.path.join(root_dir,name,a),
                            os.path.join(valid_dir,name,a))

    print('Train and validate set successfully split!')


def data_count(root_dir, chosen=''):
    """
    To count data
    :param root_dir:
    :param chosen:
    :return:
    """

    all_list = os.listdir(root_dir)
    # for name in chosen:
    #     # To remove the test set
    #     all_list.remove(name)


    # a = set([i[0] for i in all_list])
    a = {'1': 0, '2': 0, '5': 0, '8': 0}
    b = a.copy()
    for i in all_list:
        a[i[0]] += len(os.listdir(os.path.join(root_dir, i)))
        b[i[0]] += 1
    print('Total patches of all kinds:{}'.format(a))
    print('Total slides of all kinds:{}'.format(b))


if __name__ == "__main__":

    # coors = read_xml(file_path=note_path,
    #                  mask_level=0)
    # print(coors)

    # thumbnail,mask = make_mask(slide_path=slide_path,
    #                            note_path=note_path,
    #                            mask_level=4,
    #                            mask_path=mask_path,
    #                            map_path=map_path)
    # #
    # img_plot(image=thumbnail,
    #          image_name=slide_path)

    # make_patch(slide_path=slide_path,
    #            mask_path=mask_path,
    #            patch_path=patch_path,
    #            patch_size=256,
    #            raw_level=3,
    #            mask_level=4,
    #            single_file=single_file)

    # region_plot(slide_path=slide_path,
    #             coor=15000,
    #             level=2,
    #             patch_size=256)

    # data_generate(root=ROOT_DIR,
    #               note_root=ANOTE_DIR,
    #               map_path=MAP_DIR,
    #               mask_path=MASK_DIR,
    #               patch_path=PATCH_DIR,
    #               raw_level=3, mask_level=4,
    #               p_size=256)

    # # '/temp/zhaiyupeng/tianjing/v1/mask/'
    # break_restore(root_dir='/data/images/pathology/temp/TjTiff/',
    #               obj_dir='/temp/zhaiyupeng/tianjing/v1/l3/')
    # '503-537838-2'
    chosen_file = ['502-518934-1','503-537838-5','227-534248-3','224-533214-4']
    # ,'230-534771-5','503-537838-2'
    # '813-539095-6', '821-539288-7', '828-539369-11',''

    # data_divide(root_dir='/temp/zhaiyupeng/tianjing/v1/l3/',
    #             chosen=chosen_file,
    #             test_dir='/temp/zhaiyupeng/tianjing/v1/data_3/test/',
    #             train_dir='/temp/zhaiyupeng/tianjing/v1/data_3/train/',
    #             valid_dir='/temp/zhaiyupeng/tianjing/v1/data_3/validate/',
    #             )

    data_count(root_dir='/temp/zhaiyupeng/tianjing/v1/data_3/train/',
               chosen=chosen_file)