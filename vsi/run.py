from utils import get_config, get_files
from openVSI import VSIReader
import matplotlib.pyplot as plt

if __name__ == '__main__':

    root_path = get_config('config.yaml')
    file_paths = get_files(directory=root_path, keyword='.v si')
    a_file = file_paths[1]
    print('Path of the current .vsi file: {}'.format(a_file))

    reader = VSIReader()

    reader.load_vsi(file_path=a_file)

    img_region = reader.read_region(location=(1200,1000),
                                    series=4,
                                    size=(128,128))

    img_thumbnail = reader.get_thumbnail(size=(500, 500))

    print('Size of the current series: {}'.format(reader.level_dimension(series=5)))

    print('Rate of down sampling: {}'.format(reader.level_downsamples(series=5)))
    reader.close()

    plt.figure('Thumbnail')
    plt.imshow(img_thumbnail)

    plt.figure('Region')
    plt.imshow(img_region)
    plt.show()
