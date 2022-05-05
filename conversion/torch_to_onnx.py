import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import torch

import model_io
from models import UnetAdaptiveBins


def torch_to_onnx(dataset='nyu', device='cuda:0'):
    if dataset == 'nyu':
        min_depth = 1e-3
        max_depth = 10
        saving_factor = 1000  # used to save in 16 bit
        model = UnetAdaptiveBins.build(n_bins=256, min_val=min_depth, max_val=max_depth)
        pretrained_path = "../pretrained/AdaBins_nyu.pt"
    elif dataset == 'kitti':
        min_depth = 1e-3
        max_depth = 80
        saving_factor = 256
        model = UnetAdaptiveBins.build(n_bins=256, min_val=min_depth, max_val=max_depth)
        pretrained_path = "../pretrained/AdaBins_kitti.pt"
    else:
        raise ValueError("dataset can be either 'nyu' or 'kitti' but got {}".format(dataset))

    model, _, _ = model_io.load_checkpoint(pretrained_path, model)
    model.eval()
    model = model.to(device)

    dummy_input = torch.randn(1, 3, 640, 480, device=device)
    dummy_output = model(dummy_input)

    torch.onnx.export(model, dummy_input, "adabins.onnx", 
    verbose=True, example_outputs=dummy_output)


if __name__ == "__main__":
    torch_to_onnx()