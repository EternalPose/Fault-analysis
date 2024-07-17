import numpy as np
import torch
from sine_utils import run_sine_segmentation
import plotly.graph_objects as go

raw_data = torch.load('/home/user/codes/POWERODE/ode/LatentSegmentedODE-main/LatentSegmentedODE-main/SineWave/Data/Train/sine_train_10000_100_7_0.025')
print(1)