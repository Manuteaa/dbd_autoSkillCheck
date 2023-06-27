import glob
import os
import shutil
import tqdm


def move_images(source_folder, destination_folder, n):
    files = glob.glob(os.path.join(source_folder, "*.*"))
    files += glob.glob(os.path.join(source_folder, "*", "*.*"))
    files.sort()

    files_chunks = [files[i:i+n] for i in range(0, len(files), n)]

    # iterate over files_chunks with a tqdm progress bar
    for files_chunk in tqdm.tqdm(files_chunks):
        files_to_remove = files_chunk[:n-1]
        for file in files_to_remove:
            file_dest = os.path.join(destination_folder, os.path.basename(file))
            shutil.move(file, file_dest)


if __name__ == '__main__':
    source_folder = 'dataset/0'
    destination_folder = 'dataset/0_removed'

    if not os.path.exists(destination_folder):
        os.mkdir(destination_folder)

    move_images(source_folder, destination_folder, 5)
