import numpy as np
import os
import shutil
import random

import cv2
import matplotlib.pyplot as plt

from xml.etree.ElementTree import parse
import openslide

from tqdm import tqdm


def read_xml(file_path, mask_level):
    """
    To read coordinates in xml files
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
    :return:
    """
    if not os.path.exists(mask_path):
        os.mkdir(mask_path)
    if not os.path.exists(map_path):
        os.mkdir(map_path)

    # name of the file excluding the suffix
    file_name = slide_path.split('/')[-1].split('.')[0]
    # slide loading
    slide = openslide.OpenSlide(slide_path)
    slide_thumbnail = np.array(slide.get_thumbnail(slide.level_dimensions[mask_level]))

    # xml loading
    coors_list = read_xml(os.path.join(note_path, file_name + '.xml'), mask_level)

    # draw boundary of annotation in the thumbnail
    for coors in coors_list:
        # print(np.array([coors]))
        # -1: 绘制所有  5: 轮廓粗细
        cv2.drawContours(slide_thumbnail, np.array([coors]), -1, (0,255,0), 10)
    # save thumbnail with the annotaion
    cv2.imwrite(os.path.join(map_path, file_name + '.png'), slide_thumbnail)

    # draw the area covered by annotation and return a black-and-white image
    region_mask = np.zeros(slide.level_dimensions[mask_level][::-1]) # To transpose width and height
    for coors in coors_list:
        cv2.drawContours(region_mask, np.array([coors]), -1, 255, -1)
    cv2.imwrite(os.path.join(mask_path, file_name + '.png'), region_mask)

    return slide_thumbnail, region_mask


class MultiScale():

    def __init__(self, slide_path, mask_path, map_path, patch_path, p_size):

        # self.slide_list = os.listdir(slide_dir)
        self.slide_path = slide_path
        self.mask_path = mask_path
        self.map_path = map_path
        self.patch_path = patch_path
        self.p_size = p_size

        self.slide = None

        if not os.path.exists(self.patch_path):
            os.makedirs(self.patch_path)

    def _load(self, slide_path, file_name):

        slide = openslide.OpenSlide(slide_path)
        self.slide = slide
        slide_mask = cv2.imread(os.path.join(self.mask_path, file_name + '.png'), 0)
        slide_map = cv2.imread(os.path.join(self.map_path, file_name + '.png'))

        return slide, slide_mask, slide_map

    def _calulate(self, _1_level, _2_level, mask_level):

        width, height = np.array(self.slide.level_dimensions[_1_level]) // self.p_size

        stride = int(self.p_size / (2**(mask_level - _1_level)))
        zoom_ratio = self.p_size * (2 ** _1_level)
        micro_count = 2 ** (_1_level - _2_level)

        return stride, zoom_ratio, micro_count, (width, height)

    def _micro(self, level):
        pass


    def make_patch(self, file_name, _1_level, _2_level, mask_level, thr):
        """
        To generate patch using mask and threshold
        :return:
        """
        if not os.path.exists(os.path.join(self.patch_path, file_name)):
            os.mkdir(os.path.join(self.patch_path, file_name))

        micro_dir = os.path.join(self.patch_path, file_name + '_l' + str(_2_level))
        if not os.path.exists(micro_dir):
            os.mkdir(micro_dir)

        slide, mask, map= self._load(os.path.join(self.slide_path, file_name + '.tif'), file_name)
        # print('Size of slide:{}'.format(slide.level_dimensions[_1_level])

        # stride represents the patch_size in the slide map
        stride, zoom, micro_count, (width, height)  = self._calulate(_1_level, _2_level, mask_level)
        print('Patches on width:{}, Patches on height:{}'.format(width, height))
        total = width * height
        all_cnt = 0
        t_cnt = 0

        for i in range(width):
            for j in range(height):
                pixel_sum = mask[stride * j: stride * (j + 1),
                            stride * i: stride * (i + 1)].sum()
                nonzero_ratio = pixel_sum / (stride* stride* 255)

                if (nonzero_ratio >= thr):
                    t_cnt += 1
                    patch_name = os.path.join(self.patch_path, file_name, str(i) + '_' + str(j) + '.png')

                    patch = slide.read_region((zoom*i, zoom*j),
                                              _1_level,
                                              (self.p_size, self.p_size))
                    patch.save(patch_name)
                    cv2.rectangle(map,
                                  (stride * i, stride * j),
                                  (stride * (i + 1), stride * (j + 1)),
                                  (0, 0, 255),
                                  5)

                    for m_i in range(micro_count):
                        for m_j in range(micro_count):
                            folder = os.path.join(micro_dir, str(i) + '_' + str(j))
                            if not os.path.exists(folder):
                                os.mkdir(folder)
                            patch_name = os.path.join(folder, '_' + str(m_i) + '_' + str(m_j) + '.png')
                            patch = slide.read_region((zoom*i + m_i*self.p_size, zoom*j + m_j*self.p_size),
                                                      _2_level,
                                                      (self.p_size, self.p_size))
                            patch.save(patch_name)

                all_cnt += 1
                print('\rProcess: %.3f%%,  All: %d, Extracted: %d'
                      % ((100. * all_cnt / total), all_cnt, t_cnt), end="")

        cv2.imwrite(self.map_path + '_' + file_name + '.png', map)

        return t_cnt


if __name__ == '__main__':

    # slide = ['/data/images/pathology/temp/TjTiff/519-539271-10.tif',
    root = '/data/images/pathology/temp/TjTiff/'
    # all = os.listdir(root)
    all = ['519-539271-10.tif','518-539268-1.tif','507-538853-21.tif','507-538853-22.tif','519-539271-1.tif',
           '189-502217-3.tif','224-533214-3.tif','207-518542-2.tif','205-517631-10.tif','157-480417-2.tif',
           '810-538970-14.tif','804-538458-17.tif','823-539321-10.tif','811-539000-10.tif','818-539268-17.tif']

    # for a in tqdm(all, total=len(all)):
    #     if a.split('.')[-1] == 'tif':
    #         make_mask(slide_path=os.path.join(root, a),
    #                   note_path='/data/images/pathology/temp/TjAnnotation/',
    #                   mask_level=4,
    #                   mask_path='/temp/zhaiyupeng/tianjing/v2/mask',
    #                   map_path='/temp/zhaiyupeng/tianjing/v2/map',)

    ms = MultiScale(slide_path=root,
                    mask_path='/temp/zhaiyupeng/tianjing/v2/mask/',
                    map_path='/temp/zhaiyupeng/tianjing/v2/map/',
                    patch_path='/temp/zhaiyupeng/tianjing/v2/patch/',
                    p_size=256)
    for a in tqdm(all, total=len(all)):
        ms.make_patch(file_name=a.split('.')[0],
                      _1_level=3,
                      _2_level=1,
                      mask_level=4,
                      thr=0.9)