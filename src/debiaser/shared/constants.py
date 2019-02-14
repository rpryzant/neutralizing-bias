"""
just a lil global variable so everybody knows whether
we have a gpu or not

TODO maybe turn into some kind of constants file? 
"""

import torch


CUDA = (torch.cuda.device_count() > 0)


