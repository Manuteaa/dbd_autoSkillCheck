import glob
import os

from dbd.utils.dataset_utils import delete_similar_images, delete_consecutive_images


if __name__ == '__main__':
    source_folder = '<FOLDER>/0'
    assert os.path.isdir(source_folder)

    files = glob.glob(os.path.join(source_folder, "*.*"))
    files.sort()
    delete_consecutive_images(files, 100)

    files = glob.glob(os.path.join(source_folder, "*.*"))
    files.sort()
    delete_similar_images(files)
