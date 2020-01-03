from __future__ import print_function
import os
import cv2
from tqdm import tqdm
import numpy as np
import javabridge as jb
import bioformats as bf
from ruamel import yaml
import matplotlib.pyplot as plt


class VSIReader():

    def __init__(self):

        jb.start_vm(class_path=bf.JARS)
        self.omexml = bf.metadatatools.createOMEXMLMetadata()
        self.ImageReader = bf.formatreader.make_image_reader_class()
        self.reader = self.ImageReader()
        self.reader.setMetadataStore(self.omexml)

    def load_vsi(self, file_path):
        """
        Load a .vsi file
        Usage:
            >>> get_vsi('f.vsi')
        """

        self.reader.setId(file_path)
        self.series_count = self.reader.getSeriesCount()
        print('Series count: {}'.format(self.series_count))

    def close(self):
        """
        kill the java vm
        """
        jb.kill_vm()

    def read_region(self, location, series, size):
        """
        Read a certain area region of the vsi and return a RGB image
        Usage:
            >>> read_region((0,0),6,(1024,1024))
            This will read a region consist of 1024x1024 pixels
            with offset (0,0) and series 6 of the input .vsi image
        """

        print('Series: {}'.format(series))
        self.reader.setSeries(series)  # to determine which series, series 0 is the max

        img_count = self.reader.getImageCount()
        print('Images count in a series: {}'.format(img_count))
        #  count of images in a series, usually is 1

        if img_count is 1:
            offset_x = location[0]  # offset x
            offset_y = location[1]  # offset y
            x = size[0]  # size x
            y = size[1]  # size y
            total_x = self.reader.getSizeX()  # image total width
            total_y = self.reader.getSizeY()  # image total height
            c = self.reader.getSizeC()  # count of color planes of the image
            print('Image total size: {}x{}'.format(total_x, total_y))
            print('Region size: {}x{}'.format(x, y))
            if offset_x + x <= total_x and offset_x + y <= total_y:
                try:
                    data = self.reader.openBytesXYWH(0, offset_x, offset_y, int(x), int(y))
                    # print(len(data))

                    img = np.reshape(data, (y, x, c))  # img: c * (x * y)
                    img_merged = cv2.merge([img[:, :, 0], img[:, :, 1], img[:, :, 2]])

                except Exception as err:
                    print(err)
                    img_merged = None
            else:
                print('Input size has exceeded!')
                jb.kill_vm()

        else:
            img_merged = None

        return img_merged

    def get_thumbnail(self, size):
        """
        Return a thumbnail image according to the given size of .vsi with a format of RGB
        Notice that you'd better not to try to get a enormous thumbnail, it would very slow
        Usage:
            >>> get_thumbnail((1024,1024))
            >>> get_thumbnail([1024,1024])
            This will return a thumbnail image with the size of 1024x1024
        """
        for s in range(self.series_count-1,-1,-1):
            self.reader.setSeries(s)
            img_count = self.reader.getImageCount()
            #  count of images in a series, usually is 1
            if img_count == 1:
                total_x = self.reader.getSizeX()  # image total width
                total_y = self.reader.getSizeY()  # image total height
                c = self.reader.getSizeC()  # count of color planes of the image
                if total_x >= size[0] and total_y >= size[1]:
                    data = self.reader.openBytesXYWH(0, 0, 0, int(total_x), int(total_y))
                    img = np.reshape(data, (total_y, total_x, c))  # img: c * (x * y)
                    img_merged = cv2.merge([img[:, :, 0], img[:, :, 1], img[:, :, 2]])
                    img_resized = cv2.resize(img_merged,(size[1],size[0]))
                    break
                else:
                    continue
            else:
                img_resized = None

        return img_resized

    def level_dimension(self, series):
        """
        Return the size of the image with a certain series
        Usage:
            >>> level_dimension(5)
            This will return the size of series 5
        """
        self.reader.setSeries(series)
        img_count = self.reader.getImageCount()
        if img_count == 1:
            total_x = self.reader.getSizeX()  # image total width
            total_y = self.reader.getSizeY()  # image total height
            m,n = total_x,total_y
            return [m,n]
        else:
            return None

    def level_downsamples(self, series):
        """
        Return the rate of down sampling
        Usage:
            >>> level_downsamples(5)
        """
        self.reader.setSeries(0)
        x_0 = self.reader.getSizeX()
        self.reader.setSeries(series)
        x = self.reader.getSizeX()

        return x_0 / x


if __name__ == '__main__':

    root_path = get_config('config.yaml')

    file_paths = get_files(directory=root_path, keyword='.vsi')
    a_file = file_paths[0]
    print(a_file)

    ll = VSIReader()
    ll.load_vsi(file_path=a_file)
    # img = ll.read_region(location=(1200,1000),
    #                      series=4,
    #                      size=(128,128))
    img = ll.get_thumbnail(size=(500,500))
    print(ll.level_dimension(series=5))
    print(ll.level_downsamples(series=5))

    plt.imshow(img)
    plt.show()


