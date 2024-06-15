import cv2
import torch
from basicsr.utils import img2tensor
from ldm.util import resize_numpy_image
from PIL import Image
from torch import autocast

def get_cond_canny(opt, cond_image, cond_inp_type='image', cond_model=None):
    if isinstance(cond_image, str):
        canny = cv2.imread(cond_image)
    else:
        canny = cv2.cvtColor(cond_image, cv2.COLOR_RGB2BGR)
    canny = resize_numpy_image(canny, max_resolution=opt.max_resolution, resize_short_edge=opt.resize_short_edge)
    opt.H, opt.W = canny.shape[:2]
    if cond_inp_type == 'canny':
        canny = img2tensor(canny)[0:1].unsqueeze(0) / 255.
        canny = canny.to(opt.device)
    elif cond_inp_type == 'image':
        canny = cv2.Canny(canny, 100, 200)[..., None]
        canny = img2tensor(canny).unsqueeze(0) / 255.
        canny = canny.to(opt.device)
    else:
        raise NotImplementedError

    return canny

def get_cond_depth(opt, cond_image, cond_inp_type='image', cond_model=None):
    if isinstance(cond_image, str):
        depth = cv2.imread(cond_image)
    else:
        depth = cv2.cvtColor(cond_image, cv2.COLOR_RGB2BGR)
    depth = resize_numpy_image(depth, max_resolution=opt.max_resolution, resize_short_edge=opt.resize_short_edge)
    opt.H, opt.W = depth.shape[:2]
    if cond_inp_type == 'depth':
        depth = img2tensor(depth).unsqueeze(0) / 255.
        depth = depth.to(opt.device)
    elif cond_inp_type == 'image':
        depth = img2tensor(depth).unsqueeze(0) / 127.5 - 1.0
        depth = cond_model(depth.to(opt.device)).repeat(1, 3, 1, 1)
        depth -= torch.min(depth)
        depth /= torch.max(depth)
    else:
        raise NotImplementedError

    return depth
