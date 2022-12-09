import numpy as np
import cv2
import mss
import pyautogui

width, height = pyautogui.size()
object_size_h = height // 6
object_size_w = width // 6
object_size = max(object_size_w, object_size_h)

monitor = {"top": height // 2 - object_size // 2,
           "left": width // 2 - object_size // 2,
           "width": object_size,
           "height": object_size}

with mss.mss() as sct:
    i = 0
    while True:
        screenshot = np.array(sct.grab(monitor))
        cv2.imwrite("E:/temp/dataset/vid/e{}.png".format(i), screenshot)
        i += 1
        # cv2.imshow('screen', screenshot)
        # cv2.waitKey(1)


# images = glob("E:/temp/dataset/0_full/*.png")
# i = 0
# for image in images:
#     img = torchvision.io.read_image(image)
#     img = tff.center_crop(img, [224, 224])
#
#     img_np = torch.permute(img, [1, 2, 0]).numpy()
#     img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
#     cv2.imwrite("E:/temp/dataset/0/{}.png".format(i), img_np)
#     # cv2.imshow("", img_np)
#     # cv2.waitKey(0)
#     i += 1

