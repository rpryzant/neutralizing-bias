"""
just a lil global variable so everybody knows whether
we have a gpu or not
"""

import torch


CUDA = (torch.cuda.device_count() > 0)


