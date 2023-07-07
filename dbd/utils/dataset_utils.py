import os
import tqdm
import cv2
import numpy as np


def delete_similar_images(files):
    similar_frames = 0
    for i in tqdm.tqdm(range(len(files)-1)):
        im1 = files[i]
        im2 = files[i+1]

        image1 = cv2.imread(im1)
        image2 = cv2.imread(im2)

        diff = np.abs(image1.astype(np.float32) - image2.astype(np.float32))
        diff = (diff[:, :, 0] + diff[:, :, 1] + 10.0 * diff[:, :, 2]) / 3.0  # add more weight to red channel
        diff = np.mean(diff) / 255.

        # print(diff, im1, im2)
        if diff < 0.01:
            os.remove(im1)
            similar_frames += 1
            # print("deleting {} with score {}".format(im1, diff))

    print("deleted {} similar frames".format(similar_frames))


def delete_consecutive_images(files, n):
    files_chunks = [files[i:i+n] for i in range(0, len(files), n)]

    # iterate over files_chunks with a tqdm progress bar
    for files_chunk in tqdm.tqdm(files_chunks):
        files_to_remove = files_chunk[:n-1]
        for file in files_to_remove:
            os.remove(file)

