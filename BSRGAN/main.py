import torch
import cv2
import numpy as np

from network_rrdbnet import RRDBNet as net

model_path = "../models/BSRGAN_x4.pth"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()
model = net(in_nc=3, out_nc=3, nf=64, nb=23, gc=32, sf=4)
model.load_state_dict(torch.load(model_path), strict=True)
model.eval()
for k, v in model.named_parameters():
    v.requires_grad = False
model = model.to(device)
torch.cuda.empty_cache()

img = cv2.imread('../images/input/input.jpg', cv2.IMREAD_COLOR).astype(np.float32) / 255.
img = np.transpose(img if img.shape[2] == 1 else img[:, :, [2, 1, 0]], (2, 0, 1))  # HCW-BGR to CHW-RGB
img = torch.from_numpy(img).float().unsqueeze(0)

output = model(img)

output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
if output.ndim == 3:
    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))  # CHW-RGB to HCW-BGR
output = (output * 255.0).round().astype(np.uint8)  # float32 to uint8
cv2.imwrite(f'../images/output/output-BSRGAN.png', output)
