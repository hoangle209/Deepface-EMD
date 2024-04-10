import sys
sys.path.insert(1, ".")

from deepface_emd.models import DeepfaceEMD
from omegaconf import OmegaConf
import glob
import cv2
from skimage import io

PATH = "face"

if __name__ == "__main__":
    cfg = OmegaConf.load("config/default.yaml")
    model = DeepfaceEMD(cfg)
    images = sorted(glob.glob(f"{PATH}/*.jpg"))
    im = io.imread(images[11])

    allign_im = model.alligning_faces(im)

    cv2.imshow("img", allign_im)


