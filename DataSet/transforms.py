from __future__ import absolute_import

from torchvision.transforms import *
from PIL import Image
import random
import math
import torch

class CovertBGR(object):
    def __init__(self):
        pass

    def __call__(self, img):
        r, g, b = img.split()
        img = Image.merge("RGB", (b, g, r))
        return img


def _get_pixels(per_pixel, rand_color, patch_size, dtype=torch.float32, device = 'cuda'):

    if per_pixel:
        return torch.empty(patch_size, dtype=dtype).normal_().to(device=device)
    elif rand_color:
        return torch.empty((patch_size[0], 1, 1), dtype = dtype).normal_().to(device=device)
    else:
        return torch.zeros((patch_size[0], 1, 1), dtype = dtype, device= device)





class RandomErasing:

    def __init__(self,
                 probability = 0.5, sl = 0.02, sh = 1/3, min_aspect =0.3,
                 mode = 'const', device='cuda'):
        self.probability = probability
        self.sl = sl
        self.sh = sh
        self.min_aspect = min_aspect
        mode = mode.lower()
        self.rand_color = False
        self.per_pixel = False
        if mode == 'rand':
            self.rand_color = True
        elif mode == 'pixel':
            self.per_pixel = True
        elif mode == 'mean':
            self.per_mean = [104/255.0, 117/255.0, 128/255.0]
        else:
            assert not mode or mode == 'const'
        self.device = device
        self.mode = mode


    def _erase(self, img, chan, img_h, img_w, dtype):
        if random.random() > self.probability:
            return
        area = img_h * img_w
        for attempt in range(100):

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.min_aspect, 1/self.min_aspect)
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area/aspect_ratio)))

            if w < img_w and h<img_h:
                top = random.randint(0, img_h - h)
                left = random.randint(0, img_w - w)
                if self.mode == 'mean':
                    img[0, top:top+h, left:left+w] = self.per_mean[0]
                    img[1, top:top+h, left:left+w] = self.per_mean[1]
                    img[2, top:top+h, left:left+w] = self.per_mean[2]
                else:
                    img[:, top:top+h, left:left+w] = _get_pixels(
                        self.per_pixel, self.rand_color, (chan, h, w),
                        dtype = dtype, device = self.device
                    )


    def __call__(self, input):
        if len(input.size()) == 3:
            self._erase(input, input.size(0), input.size(1), input.size(2), input.dtype)
        else:
            batch_size, chan, img_h, img_w = input.size()
            for i in range(batch_size):
                self._erase(input[i], chan, img_h, img_w, input.dtype)

        return input
