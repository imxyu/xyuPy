import torch
from unet3d.model import UNet3D
import torch.nn as nn
model = UNet3D(1,1,final_sigmoid=False)
model = model.cuda()

dummy_input = torch.randn(1,1,20,256,256)
out = model(dummy_input.cuda())