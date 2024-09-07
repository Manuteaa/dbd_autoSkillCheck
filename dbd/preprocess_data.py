import glob
import os

from dbd.utils.dataset_utils import delete_similar_images, delete_consecutive_images


if __name__ == '__main__':
    source_folder = 'dataset_prediction/4'
    assert os.path.isdir(source_folder)

    files = glob.glob(os.path.join(source_folder, "*.*"))
    files += glob.glob(os.path.join(source_folder, "*", "*.*"))
    files.sort()

    # delete_consecutive_images(files, 2)
    # delete_similar_images(files)
