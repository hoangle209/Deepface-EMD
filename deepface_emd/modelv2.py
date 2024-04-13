import torch
import torchvision.transform as T

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

        if fm == "arcface_iresnet" or fm == "arcface_iresnet_upng":
            self.facenet = iresnet()
            self.facenet.load_state_dict(state_dict)
            if self.level == 8:
                self.embed_key = "embedding_88"
                self.avg_pool_key = "adpt_pooling_88"

            if fm == "arcface_iresnet_upng":
                self.upng_norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        else:
            logger.warning(f"model type {self.cfg.face_net.type} is not supported!!!")
            exit(0)
        
        self.fea_key = 'fea'
        self.facenet = self.facenet.to(torch.device(self.device))
        self.facenet.eval()
    
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

        if fm == "arcface_iresnet":
            batch = resize(batch, (112, 112))
            batch = (batch - 0.5) / 0.5 # convert to range(-1, 1)
        elif fm == "arcface_iresnet_upng":
            batch = resize(batch, (112, 112))
            batch = self.upng_norm(batch)

        return batch.to(self.device)
    
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



