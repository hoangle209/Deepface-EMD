import torch
from torchvision import transforms as T

from collections import OrderedDict
import cv2
import numpy as np
from kornia.color import bgr_to_grayscale
from kornia.geometry.transform import resize

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
        self.fea_key = 'fea'
        self.facenet.eval()


    def preprocess(self, batch):
        """When using kornia, batch is assert with shape b, c, w, h
        and in range(0, 1)

        Parameters:
        -----------
            batch: Union[list, ndarray]
        """
        if not isinstance(batch, torch.Tensor):
            if isinstance(batch, list):
                batch = [i.transpose[None] for i in batch]
                batch = np.concatenate(batch, axis=0)
            elif isinstance(batch, np.ndarray):
                if len(batch.shape) == 3:
                    batch = batch[None]

            batch = batch.tranpose(0, 3, 1, 2)
            batch = batch / 255. # convert to range(0,1)
            batch = torch.from_numpy(batch).float()

        fm = self.cfg.face_net.type

        if fm == "arcface":
            batch = bgr_to_grayscale(batch)
            batch = resize(batch, (128, 128))
        elif fm == "cosface":
            pass
            
        batch = (batch - 0.5) / 0.5 # convert to range (-1, 1)
        return batch 


    def extract_feat(self, batch):
        batch = self.preprocess(batch)
        out = self.facenet(batch)

        anchor = out[self.embed_key]
        anchor_center = out[self.fea_key]
        avgpool_bank_center_query = out[self.avg_pool_key].squeeze(-1).squeeze(-1)

        return anchor, anchor_center, avgpool_bank_center_query


    def find_smilarities(self, queries, targets, topK=5):
        """
        Parameters:
        -----------
            queries
        """


    def __call__(self, queries, targets):
        pass

