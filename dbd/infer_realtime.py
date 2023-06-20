import numpy as np
import cv2
import mss
import torch

from dbd.utils.frame_grabber import get_monitor_attributes

if __name__ == '__main__':
    pass
    # Make new dataset folder, where we save the frames
    # timestr = time.strftime("%Y%m%d-%H%M%S")
    # os.mkdir(os.path.join("dataset", timestr))
    #
    # # Get monitor attributes
    # monitor = get_monitor_attributes()
    #
    # with mss.mss() as sct:
    #     i = 0
    #
    #     # Infinite loop
    #     while True:
    #         screenshot = np.array(sct.grab(monitor))
    #         cv2.imwrite(os.path.join("dataset", "frame_{}.png".format(i)), screenshot)
    #         i += 1
