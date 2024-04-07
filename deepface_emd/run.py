import torch
from torchvision import transforms as T

from collections import OrderedDict
import cv2
import numpy as np

from third_parties.deepface.resnet import resnet_face18
from third_parties.deepface.cosnet import sphere

from deepface_emd.utils.emd import emd_similarity
from deepface_emd.utils import get_pylogger
logger = get_pylogger()

class DeepfaceEMD:
    def __init__(self, config) -> None:
        self.cfg = config
        self.load_model()
    
    def load_model(self):
        checkpoint = self.cfg.face_net.weight
        state_dict = torch.load(checkpoint)
        
        if self.cfg.face_net.type == "arcface":
            self.facenet = resnet_face18(False, use_reduce_pool=True)
            _state_dict = OrderedDict()
            for k, v in state_dict.items():
                _state_dict[k[7:]] = v # remove module.
            self.facenet.load_state_dict(_state_dict)
            self.embed_key = 'embedding_44'
            self.avg_pool_key = 'adpt_pooling_44'
        
        elif self.cfg.face_net.type == "cosface":
            self.facenet = sphere()
            self.facenet.load_state_dict(state_dict)
            self.embed_key = 'embedding'
            self.avg_pool_key = 'adpt_pooling'
        
        else:
            logger.warning(f"model type {self.cfg.face_net.type} is not supported. Using arcface module as default !")

    def preprocess(query_img, shape, fm='arcface'):
        # TODO, arcface recieve gray image, convert image to range (-1, 1)

        if fm == 'arcface':
            img = cv2.resize(img, shape)
            img = img.reshape((128,128,1))
            img = img.transpose((2, 0, 1))
            img = img.astype(np.float32, copy=False)
        elif fm == 'cosface':
            return img
        else:
            print('No face model found!!')
            exit(0)

        return torch.from_numpy(img).float()


    def __call__(self, queries, targets):
        pass

