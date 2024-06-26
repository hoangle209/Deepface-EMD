import sys
sys.path.insert(0, ".")

from deepface_emd.modelv2 import DeepfaceEMD
from omegaconf import OmegaConf
import glob
import cv2
from skimage import io

# PATH = "/home/hoang/Downloads/MQ/face/test_face_recognize/face"
# PATH = "/home/hoang/Downloads/data/ND_KhuonMat"
PATH = "data/face"

if __name__ == "__main__":
    images = sorted(glob.glob(f"{PATH}/*.jpg"))
    cfg = OmegaConf.load("config/default.yaml")
    model = DeepfaceEMD(cfg)

    x = 99
    y = 10
    z = 95

    # images[x] = PATH + "/59_20230617_042222" + ".jpg"
    # images[y] = PATH + "/59_20230617_042218" + ".jpg"
    # images[z] = PATH + "/59_20230903_105826" + ".jpg"

    print(images[x])
    print(images[y])
    # print(images[z])

    im1 = io.imread(images[x])
    im2 = io.imread(images[y])
    # im3 = io.imread(images[z])

    model.find_similarities([im1], [im2])



