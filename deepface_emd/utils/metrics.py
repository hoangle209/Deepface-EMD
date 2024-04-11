import torch
import numpy as np

def angular_distance(cosine_smilarity):
    if isinstance(cosine_smilarity, torch.Tensor):
        angular_distance = cosine_smilarity.acos() / np.pi
    elif isinstance(cosine_smilarity, np.ndarray):
        angular_distance = np.arccos(cosine_smilarity) / np.pi
    
    angular_similarity = 1 - angular_distance
    return angular_similarity