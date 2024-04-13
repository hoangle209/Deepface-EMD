import sys
sys.path.insert(1, ".")

from deepface_emd.models import DeepfaceEMD
from omegaconf import OmegaConf
import glob
import cv2
from skimage import io
import torch

PATH = "face"

if __name__ == "__main__":
    from third_parties.deepface import iresnet

    model = iresnet(100)
    weight = torch.load("weights/face.r100.arc.unpg.pt", map_location="cpu")
    model.load_state_dict(weight)


