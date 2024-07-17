import os
import sys
import argparse
from pathlib import Path
from utils import gpu_f, to_np
import matplotlib.pyplot as plt

import numpy as np
import torch

from sine_utils import run_sine_segmentation1

sys.path.append(os.path.abspath('..'))
from models import LatentODEBuilder

np.random.seed(2547)

device = torch.device('cuda:0')

# Disables autograd, reduces memory usage
torch.autograd.set_grad_enabled(False)

des = "Evaluate trained model on Sine Wave data."
parser = argparse.ArgumentParser(description=des)
parser.add_argument("--model_file", type=str, default='/home/user/codes/POWERODE/ode/LatentSegmentedODE-main/LatentSegmentedODE-main/SineWave/Models/sine_lode_sine_train_10000_100_7_0.025_2024-01-11-01-15-21')
parser.add_argument("--data_file", type=str, default='/home/user/codes/POWERODE/ode/LatentSegmentedODE-main/LatentSegmentedODE-main/SineWave/Data/Test/sine_test_100_7_0.025_3')
parser.add_argument("--n_samp", type=int, default=100)
parser.add_argument("--min_seg_len", type=int, default=10)
parser.add_argument("--K", type=float, default=200)
parser.add_argument("--n_dec", type=int, default=2)
parser.add_argument("--l_var", type=float, default=1)
args = parser.parse_args()

data_root = Path("./Data/Test")
model_root = Path("./Models")
output_root = Path("./Results")

save_data = torch.load(model_root / Path(args.model_file))

model_args = save_data['model_args']
model = LatentODEBuilder(model_args).build_model().to(device)
model.load_state_dict(save_data['model_state_dict'])


raw_data = torch.load(data_root / Path(args.data_file))
data = raw_data['data']

test_args = {
    'n_samp': args.n_samp,
    'min_seg_len': args.min_seg_len,
    'K': args.K,
    'n_dec': args.n_dec,
    'l_var': args.l_var
}

for pack in data:
    # out = to_np(model.predict(gpu_f(pack[0]), gpu_f(pack[1])))
    # fig = plt.figure() 
    # ax = fig.add_subplot(111)
    # ax.plot(pack[1], pack[0].squeeze(), c='red', alpha=0.8)
    # ax.plot(pack[1], out.squeeze(), c='orange', alpha=0.9, linestyle='--')
    pred_all, scores_all = run_sine_segmentation1(pack, model, **test_args)
    pred = pred_all[0].tolist()
    pred.insert(0, 0)
    pred.append(len(pack[0][0])-1)

    out = []
    for i in range(len(pred)-1):
        if i != len(pred)-1:
            out = out + to_np(model.predict(gpu_f(pack[0][:,pred[i]:pred[i+1]]), gpu_f(pack[1][pred[i]:pred[i+1]]))).squeeze().tolist()
        else:
            out = out + to_np(model.predict(gpu_f(pack[0][:,pred[i]:]), gpu_f(pack[1][pred[i]:]))).squeeze().tolist()
    # out = np.array(out)
    #    out = to_np(model.predict(gpu_f(pack[0]), gpu_f(pack[1])))
    fig = plt.figure() 
    ax = fig.add_subplot(111)
    ax.plot(pack[1], pack[0].squeeze(), c='red', alpha=0.8)
    ax.plot(pack[1][0:len(pack[0][0])-1].tolist(), out, c='orange', alpha=0.9, linestyle='--')

# pred_all, scores_all = run_sine_segmentation(data, model, **test_args)

# results = {
#     'pred_all': pred_all,
#     'model_path': model_root / Path(args.model_file),
#     'data_path': data_root / Path(args.data_file),
#     'test_args': test_args,
# }

# output_root.mkdir(parents=True, exist_ok=True)
# output_fn = "results_{}_{}_{}_{}".format(args.model_file, args.data_file,
#                                          args.n_samp, args.min_seg_len)

# torch.save(results, output_root / Path(output_fn))
