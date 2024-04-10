import torch
from torchvision import transforms as T

from collections import OrderedDict
import numpy as np
from kornia.color import bgr_to_grayscale
from kornia.geometry.transform import resize
from skimage import io, img_as_ubyte
import cv2

from third_parties.deepface import (
                                resnet_face18, 
                                sphere, 
                                InceptionResnetV1, 
                                iresnet50
                            )
from deepface_emd.utils.matlab_cp2tform import get_similarity_transform_for_cv2
from deepface_emd.utils.emd import emd_similarity
from deepface_emd.utils import get_pylogger
logger = get_pylogger()

class DeepfaceEMD(torch.nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.cfg = config
        self.load_model()

        if self.cfg.use_face_allignment:
            import face_alignment
            self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, 
                                                   flip_input=False,
                                                   device="cpu")

    def load_model(self):
        checkpoint = self.cfg.face_net.weight
        state_dict = torch.load(checkpoint, map_location=torch.device('cpu'))
        
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
        
        elif self.cfg.face_net.type == "arcface_iresnet":
            self.facenet = iresnet50()
            self.facenet.load_state_dict(state_dict)
            self.embed_key = 'embedding_44'
            self.avg_pool_key = 'adpt_pooling_44'
        
        else:
            logger.warning(f"model type {self.cfg.face_net.type} is not supported. Using arcface module as default !")
        self.fea_key = 'fea'
        self.facenet.eval()


    def preprocess(self, batch):
        """When using kornia, batch is assert with shape b, c, w, h
        and in range(0, 1)

        Parameters:
        -----------
            batch: Union[list, ndarray, torch.Tensor],
                list contains images in BGR format with shape HxWxC
                ndarray with shape HxWxC or bxHxWxC 
                torch.Tensor with shape bxCxHxW in range(0, 1)
        """
        if not isinstance(batch, torch.Tensor):
            if isinstance(batch, list):
                batch = [i[None] for i in batch]
                batch = np.concatenate(batch, axis=0)
            elif isinstance(batch, np.ndarray):
                if len(batch.shape) == 3:
                    batch = batch[None]

            batch = batch.transpose(0, 3, 1, 2)
            batch = batch / 255. # convert to range(0, 1)
            batch = torch.from_numpy(batch).float()

        fm = self.cfg.face_net.type

        if fm == "arcface":
            batch = bgr_to_grayscale(batch)
            batch = resize(batch, (128, 128))
        elif fm == "cosface":
            batch = resize(batch, (112, 96))
        elif fm == "facenet":
            batch = resize(batch, (160, 160))
        elif fm == "arcface_iresnet":
            batch = resize(batch, (112, 112))
            
        batch = (batch - 0.5) / 0.5 # convert to range(-1, 1)
        return batch 
    

    def extract_feat(self, batch):
        batch = self.preprocess(batch)
        out = self.facenet(batch)

        feature_bank = out[self.embed_key] 
        feature_bank_center = out[self.fea_key]
        avgpool_bank_center = out[self.avg_pool_key].squeeze(-1).squeeze(-1) 

        feature_bank = feature_bank.view(feature_bank.size(0), feature_bank.size(1), -1)
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
            # the first stage is to compute the topK candicates who are most similar
            first_stage_similarity, _, _, _ = emd_similarity(None, 
                                                             query_feature_center, 
                                                             None, 
                                                             target_feature_bank_center,
                                                             stage=0) 
            first_stage_topK_inds = torch.argsort(first_stage_similarity, descending=True)[:first_topK]
            logger.info(f"First stage similarity {first_stage_similarity}")

            # the second stage is to re-rank the candicates in the first stage
            anchor = query_feature_bank[idx]
            feature_query = query_avgpool_bank_center[idx]
            sim_avg, _, _, _ = emd_similarity(anchor, 
                                              feature_query, 
                                              target_feature_bank[first_stage_topK_inds], 
                                              target_avgpool_bank_center[first_stage_topK_inds], 
                                              stage=1, 
                                              method="apc")
            
            logger.info(f"Stage 2 sim {sim_avg}")
            if alpha < 0:
                rank_in_tops = torch.argsort(sim_avg + first_stage_similarity[first_stage_topK_inds], descending=True)[:second_topK]
            else:
                rank_in_tops = torch.argsort(alpha * sim_avg + (1.0 - alpha) * first_stage_similarity[first_stage_topK_inds], descending=True)[:second_topK]
            # rank_targets = targets[rank_in_tops.int().tolist()] # TODO, handles target here 


    def alignment(self, src_img, src_pts):
        # For 96x112
        # ref_pts = [ 
        #     [30.2946, 51.6963],
        #     [65.5318, 51.5014], 
        #     [48.0252, 71.7366],
        #     [33.5493, 92.3655],
        #     [62.7299, 92.2041]
        # ]
        # crop_size = (96, 112)

        # For 160x160
        ref_pts = [
            [52.6862,  73.8519 ],
            [109.2197, 73.5734 ], 
            [80.042,   102.4809],
            [55.9155,  131.9507],
            [104.5498, 131.7201]
        ]
        crop_size = (160, 160)
        src_pts = np.array(src_pts).reshape(5,2)

        s = np.array(src_pts).astype(np.float32)
        r = np.array(ref_pts).astype(np.float32)

        tfm = get_similarity_transform_for_cv2(s, r)
        face_img = cv2.warpAffine(src_img, tfm, crop_size)
        return face_img


    def alligning_faces(self, input):
        """
        Parameters:
        -----------
            input, ndarray,
                if ndarray, input is in RGB format
        """
        le_eye_pos = [36, 37, 38, 39, 40, 41]
        r_eye_pos  = [42, 43, 44, 45, 47, 46]

        preds = self.fa.get_landmarks_from_image(input)
        lmks = preds[0]
        le_eye_x, le_eye_y = 0.0, 0.0
        r_eye_x, r_eye_y = 0.0, 0.0
        for l_p, r_p in zip(le_eye_pos, r_eye_pos):
            le_eye_x += lmks[l_p][0]
            le_eye_y += lmks[l_p][1]
            r_eye_x  += lmks[r_p][0]
            r_eye_y  += lmks[r_p][1]
        
        le_eye_x = int(le_eye_x / len(le_eye_pos))
        le_eye_y = int(le_eye_y / len(le_eye_pos))
        r_eye_x  = int(r_eye_x  / len(r_eye_pos))
        r_eye_y  = int(r_eye_y  / len(r_eye_pos))
        nose     = (int(lmks[30][0]), int(lmks[30][1]))
        left_mo  = (int(lmks[60][0]), int(lmks[60][1]))
        ri_mo    = (int(lmks[64][0]), int(lmks[64][1]))
        final_lmks = [(le_eye_x, le_eye_y), (r_eye_x, r_eye_y), nose, left_mo, ri_mo]
        
        landmark = []
        for lmk in final_lmks:
            landmark.append(lmk[0])
            landmark.append(lmk[1])

        img = img_as_ubyte(input)[..., ::-1] # convert to BGR format
        cropped_align = self.alignment(img, landmark)

        return cropped_align


    @staticmethod
    def handle_input(input):
        if isinstance(input, list):
            input_ = []
            for im in input:
                if isinstance(im, str):
                    im = io.imread(im)
                    input_.append(im)
                else:
                    input_.append(im)
            input = input_
        
        elif isinstance(input, str):
            input = [io.imread(input)]
        else:
            input = [input]

        return input


    def run_finding_similarities(self, queries, targets):
        queries = self.handle_input(queries)
        targets = self.handle_input(targets)

        if self.cfg.use_face_allignment:
            queries = [self.alligning_faces(input) for input in queries]
            targets = [self.alligning_faces(input) for input in targets]
        
        self.find_smilarities(queries, targets)


    def forward(self, queries, targets):
        pass

