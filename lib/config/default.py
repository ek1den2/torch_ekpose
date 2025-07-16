from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from yacs.config import CfgNode as CN


_C = CN()


# common params for NETWORK
_C.MODEL = CN()

_C.MODEL.NUM_KEYPOINTS = 14
_C.MODEL.DOWNSAMPLE = 8


# testing
_C.TEST = CN()

_C.TEST.THRESH_HEATMAP =  0.15
_C.TEST.THRESH_PAF= 0.05
_C.TEST.NUM_INTERMED_PTS_BETWEEN_KEYPOINTS= 10