import torch
from torchvision import transforms as T

from collections import OrderedDict
import cv2
import numpy as np
from kornia.color import bgr_to_grayscale
from kornia.geometry.transform import resize

from third_parties.deepface import resnet_face18, sphere, InceptionResnetV1
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
        
        elif self.cfg.face_net.type == "facenet":
            self.facenet = InceptionResnetV1()
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
            batch: Union[list, ndarray, torch.Tensor]
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
            batch = resize(batch, (112, 96))
        elif fm == "facenet":
            batch = resize(batch, (160, 160))
            
        batch = (batch - 0.5) / 0.5 # convert to range (-1, 1)
        return batch 


    def extract_feat(self, batch):
        batch = self.preprocess(batch)
        out = self.facenet(batch.to(self.facenet.device))

        feature_bank = out[self.embed_key] 
        feature_bank_center = out[self.fea_key]
        avgpool_bank_center = out[self.avg_pool_key].squeeze(-1).squeeze(-1) 

        feature_bank = torch.nn.functional.normalize(feature_bank, p=2, dim=1) # ouput feature from backbone
        feature_bank_center = torch.nn.functional.normalize(feature_bank_center, p=2, dim=1) # feature after FC
        avgpool_bank_center = torch.nn.functional.normalize(avgpool_bank_center, p=2, dim=1) # GlobalAvgPooling of output feature

        return feature_bank, feature_bank_center, avgpool_bank_center


    def find_smilarities(self, queries, targets=None, first_topK=25, second_topK=5, alpha=0.3):
        """
        Parameters:
        -----------
            queries, Union[list, ndarray, torch.Tensor], 
                input query images
            targets, Union[list, ndarray, torch.Tensor],
                target images, if not given find in database
            first_topK, int,
                get the top-k most similar images
        """
        query_feature_bank, query_feature_bank_center, query_avgpool_bank_center = self.extract_feat(queries)
        target_feature_bank, target_feature_bank_center, target_avgpool_bank_center = self.extract_feat(targets)


        for idx, query_feature_center in enumerate(query_feature_bank_center):
            # the stage 0 is to compute the topK candicates who are most similar
            first_stage_similarity, _, _, _ = emd_similarity(None, 
                                                             query_feature_center, 
                                                             None, 
                                                             target_feature_bank_center,
                                                             stage=0) 
            first_stage_topK_inds = torch.argsort(first_stage_similarity, descending=True)[:first_topK]

            anchor = query_feature_bank[idx]
            feature_query = query_avgpool_bank_center[idx]
            feature_target = target_avgpool_bank_center[first_stage_topK_inds]
            sim_avg, _, _, _ = emd_similarity(anchor, 
                                              feature_query, 
                                              target_feature_bank[first_stage_topK_inds], 
                                              feature_target, 
                                              stage=1, 
                                              method="apc")
            if alpha < 0:
                rank_in_tops = torch.argsort(sim_avg + first_stage_similarity[first_stage_topK_inds], descending=True)[:second_topK]
            else:
                rank_in_tops = torch.argsort(alpha * sim_avg + (1.0 - alpha) * first_stage_similarity[first_stage_topK_inds], descending=True)[:second_topK]
            rank_targets = targets[rank_in_tops] # TODO, handles target here
            




    def __call__(self, queries, targets):
        pass

