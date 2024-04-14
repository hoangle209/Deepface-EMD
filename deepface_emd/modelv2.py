import torch
import torchvision
import torchvision.transforms as T

from kornia.geometry.transform import resize
from skimage import img_as_ubyte
import cv2
import numpy as np

from third_parties.deepface import iresnet
from deepface_emd.utils.matlab_cp2tform import get_similarity_transform_for_cv2
from deepface_emd.utils.emd import emd_similarity
from deepface_emd.utils.metrics import angular_distance
from deepface_emd.utils import get_pylogger, get_patch_location

logger = get_pylogger()

class DeepfaceEMD(torch.nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.cfg = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.load_face_recognition_net()

        if self.cfg.use_face_allignment:
            import face_alignment
            self.fa = face_alignment.FaceAlignment(
                        face_alignment.LandmarksType.TWO_D, 
                        flip_input=False,
                        device=self.device
                    )

    def load_face_recognition_net(self):
        state_dict = torch.load(self.cfg.face_net.weight, map_location=torch.device("cpu"))
        fm = self.cfg.face_net.type

        if fm == "arcface_iresnet" or fm == "arcface_iresnet_unpg":
            self.facenet = iresnet(num_layers=100)
            self.facenet.load_state_dict(state_dict)
            if self.cfg.level == 8:
                self.embed_key = "embedding_88"
                self.avg_pool_key = "adpt_pooling_88"
            else:
                self.cfg.level = 4
                self.embed_key = "embedding_44"
                self.avg_pool_key = "adpt_pooling_44"

            if fm == "arcface_iresnet_unpg":
                self.norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            else:
                self.norm = T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        else:
            logger.warning(f"model type {self.cfg.face_net.type} is not supported!!!")
            exit(0)
        
        self.fea_key = 'fea'
        self.facenet = self.facenet.to(torch.device(self.device))
        self.facenet.eval()
    
    
    def alignment(self, src_img, src_pts, crop_size=(112, 112)):
        ref_pts_96_112 = [ 
            [30.2946, 51.6963],
            [65.5318, 51.5014], 
            [48.0252, 71.7366],
            [33.5493, 92.3655],
            [62.7299, 92.2041]
        ]

        ref_pts = []
        for pts in ref_pts_96_112:
            x, y = pts
            ref_pts.append([
                x / 96  * crop_size[0], 
                y / 112 * crop_size[1]
            ])

        src_pts = np.array(src_pts).reshape(5, 2)
        s = np.array(src_pts).astype(np.float32)
        r = np.array(ref_pts).astype(np.float32)

        tfm = get_similarity_transform_for_cv2(s, r)
        face_img = cv2.warpAffine(src_img, tfm, crop_size)
        return face_img


    def alligning_face(self, input):
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
        final_lmks = [
            (le_eye_x, le_eye_y), 
            (r_eye_x, r_eye_y), 
            nose, 
            left_mo,
            ri_mo
        ]
        
        landmark = []
        for lmk in final_lmks:
            landmark.append(lmk[0])
            landmark.append(lmk[1])

        img = img_as_ubyte(input) # RGB format
        cropped_align = self.alignment(img, landmark)
        return cropped_align[None]
    

    def preprocess(self, batch):
        """When using kornia, batch is assert with shape b, c, w, h
        and in range(0, 1)

        Parameters:
        -----------
            batch: Union[list, ndarray],
                list contains images in BGR format with shape HxWxC
                ndarray with shape HxWxC or bxHxWxC 
        """
        if isinstance(batch, list):
            batch = np.concatenate(batch, axis=0)

        batch = batch.transpose(0, 3, 1, 2)
        batch = batch / 255. # convert to range(0, 1)
        batch = torch.from_numpy(batch).float()

        fm = self.cfg.face_net.type

        if fm == "arcface_iresnet" or fm == "arcface_iresnet_upng":
            batch = resize(batch, (112, 112))
        
        batch = self.norm(batch)
        return batch.to(self.device)
    

    def extract_feat(self, batch):
        batch = self.preprocess(batch)
        with torch.no_grad():
            out = self.facenet(batch)

        feature_bank = out[self.embed_key] 
        feature_bank_center = out[self.fea_key]
        avgpool_bank_center = out[self.avg_pool_key].squeeze(-1).squeeze(-1) 

        feature_bank = feature_bank.view(feature_bank.size(0), feature_bank.size(1), -1)
        feature_bank = torch.nn.functional.normalize(feature_bank, p=2, dim=1) # ouput feature from backbone
        feature_bank_center = torch.nn.functional.normalize(feature_bank_center, p=2, dim=1) # feature after FC
        avgpool_bank_center = torch.nn.functional.normalize(avgpool_bank_center, p=2, dim=1) # GlobalAvgPooling of output feature

        return feature_bank, feature_bank_center, avgpool_bank_center


    def find_smilarities(self, 
                         queries, 
                         targets=None, 
                         first_topK=25, 
                         second_topK=5, 
                         alpha=0.3,
                         method="uew"):
        """
        Parameters:
        -----------
            queries, Union[list[ndarray], ndarray],
                input query images
            targets, Union[list[ndarray], ndarray],
                target images, if not given find in database
            first_topK, int,
                get the top-k most similar images to be candicates in the first pool
            second_topK, int,
                get the top-k most similar images
            alpha, float,
                to balance first and second stage similar value
            method, str,
                weighted method

        """
        query_feature_bank, query_feature_bank_center, query_avgpool_bank_center    = self.extract_feat(queries)
        target_feature_bank, target_feature_bank_center, target_avgpool_bank_center = self.extract_feat(targets)

        for idx, query_feature_center in enumerate(query_feature_bank_center):
            # the first stage is to make a pool the topK candicates who are most similar
            first_stage_similarity, _, _, _ = emd_similarity(None, 
                                                             query_feature_center, 
                                                             None, 
                                                             target_feature_bank_center,
                                                             stage=0) 
            first_stage_topK_inds = torch.argsort(first_stage_similarity, descending=True)[:first_topK]
            logger.info(f"Stage 1 similarity {angular_distance(first_stage_similarity)}")

            # the second stage is to re-rank the candicates in the first stage
            anchor = query_feature_bank[idx]
            feature_query = query_avgpool_bank_center[idx]
            sim_avg, flows, u, v = emd_similarity(anchor, 
                                              feature_query, 
                                              target_feature_bank[first_stage_topK_inds], 
                                              target_avgpool_bank_center[first_stage_topK_inds], 
                                              stage=1, 
                                              method=method)
            
            logger.info(f"Stage 2 sim {angular_distance(sim_avg)}")
            return flows, u, v
            if alpha < 0:
                rank_in_tops = torch.argsort(sim_avg + first_stage_similarity[first_stage_topK_inds], descending=True)[:second_topK]
            else:
                rank_in_tops = torch.argsort(alpha * sim_avg + (1.0 - alpha) * first_stage_similarity[first_stage_topK_inds], descending=True)[:second_topK]
    

    def visual_flow(self, img, flow, u, v, size=(112, 112), level=4):
        img = img.transpose(2, 0, 1) / 255.0
        img = torch.from_numpy(img).float().to(self.device)

        flow = flow[0]
        patch_list = []
        weight = flow.sum(-1)
        nums = flow.shape[0]
        weight = (weight - weight.min()) / (weight.max() - weight.min())
        for index_grid in range(nums):
            index_patch=torch.argmax(flow[index_grid]).item()
            row_location, col_location, _ , _ = get_patch_location(index_patch, size[0], "arcface_iresnet", level=level)
            patch = img[:, row_location[0]:row_location[1], col_location[0]:col_location[1]]
            patch = patch * weight[index_grid]
            patch_list.append(patch)

        patch_list = torch.stack(patch_list, dim=0)
        grids = torchvision.utils.make_grid(patch_list, nrow=level, padding=0)
        grids = grids.permute(1, 2, 0).cpu().detach().numpy() * 255.0
        return grids.astype("uint8")


    def find_similarities(self, queries, targets):
        if self.cfg.use_face_allignment:
            queries = [self.alligning_face(input) for input in queries]
            targets = [self.alligning_face(input) for input in targets]
        else:
            queries = [input[None] for input in queries]
            targets = [input[None] for input in targets]
        
        flow, u, v = self.find_smilarities(queries, targets)
        grid = self.visual_flow(targets[0][0], flow, u, v, level=8)
        image = np.hstack([queries[0][0], grid, targets[0][0]])
        cv2.imshow("img", image)
        cv2.waitKey(0)



