""" global object so that everything knows whether we're on a gpu or not"""

import torch

CUDA = (torch.cuda.device_count() > 0)

