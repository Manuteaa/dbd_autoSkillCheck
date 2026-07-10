import os
import tqdm
import cv2
import numpy as np
import glob


def delete_similar_images(folder):
    files = glob.glob(os.path.join(folder, "*.*"))
    files.sort()

    similar_frames = 0
    for i in tqdm.tqdm(range(len(files)-1)):
        im1 = files[i]
        im2 = files[i+1]

        image1 = cv2.imread(im1)
        image2 = cv2.imread(im2)

        # Use int32 instead of float32 for better performance
        # Results are identical for absolute difference calculation
        diff = np.abs(image1.astype(np.int32) - image2.astype(np.int32))
        diff = (diff[:, :, 0] + diff[:, :, 1] + 10 * diff[:, :, 2]) / 3.0  # add more weight to red channel
        diff = np.mean(diff) / 255.0

        # print(diff, im1, im2)
        if diff < 0.01:
            os.remove(im1)
            similar_frames += 1
            # print("deleting {} with score {}".format(im1, diff))

    print("deleted {} similar frames".format(similar_frames))


def delete_consecutive_images(folder, n):
    files = glob.glob(os.path.join(folder, "*.*"))
    files.sort()

    files_chunks = [files[i:i+n] for i in range(0, len(files), n)]

    # iterate over files_chunks with a tqdm progress bar
    for files_chunk in tqdm.tqdm(files_chunks):
        files_to_remove = files_chunk[:n-1]
        for file in files_to_remove:
            os.remove(file)

