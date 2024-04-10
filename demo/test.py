import sys
sys.path.insert(0, ".")

# from deepface_emd.models import DeepfaceEMD
from omegaconf import OmegaConf
import glob
import cv2
from skimage import io

PATH = "/home/hoang/Downloads/MQ/face/test_face_recognize/face"

if __name__ == "__main__":
    images = sorted(glob.glob(f"{PATH}/*.jpg"))
    # cfg = OmegaConf.load("config/default.yaml")
    # model = DeepfaceEMD(cfg)

    # x = 123
    # y = 2
    # z = 10

    # print(images[x])
    # print(images[y])
    # print(images[z])

    # im1 = cv2.imread(images[x])
    # im2 = cv2.imread(images[y])
    # im3 = cv2.imread(images[z])

    im1 = io.imread("/home/hoang/Downloads/MQ/face/test_face_recognize/test_crop_warped_.jpg")
    im2 = cv2.imread("/home/hoang/Downloads/MQ/face/test_face_recognize/test_crop_warped_.jpg")
    # im2 = cv2.imread("/home/hoang/Downloads/MQ/face/test_face_recognize/test_crop.jpg")

    import numpy as np
    im = np.abs(im1-im2)

    print(np.all(im))

    # model.find_smilarities(im1, [im2, im2])



