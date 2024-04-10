import sys
sys.path.insert(0, ".")

from deepface_emd.models import DeepfaceEMD
from omegaconf import OmegaConf
import glob
import cv2
from skimage import io

PATH = "face"

if __name__ == "__main__":
    images = sorted(glob.glob(f"{PATH}/*.jpg"))
    cfg = OmegaConf.load("config/default.yaml")
    model = DeepfaceEMD(cfg)

    x = 123
    y = 2
    z = 10

    im1 = io.imread(images[x])
    im2 = io.imread(images[y])
    im3 = io.imread(images[z])

    model.run_finding_similarities(im1, [im2, im2])



