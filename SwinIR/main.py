import cv2
import numpy as np
import torch
from network_swinir import SwinIR as net

model_path = "../models/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN-with-dict-keys-params-and-params_ema.pth"
state_dict = torch.load(model_path, map_location=torch.device('cpu'))['params_ema']

model = net(upscale=4, in_chans=3, img_size=64, window_size=8, img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6], mlp_ratio=2, upsampler='nearest+conv', resi_connection='1conv')
model.load_state_dict(state_dict, strict = True)

img = cv2.imread('../images/input/input.jpg', cv2.IMREAD_COLOR).astype(np.float32) / 255.
img = np.transpose(img if img.shape[2] == 1 else img[:, :, [2, 1, 0]], (2, 0, 1))  # HCW-BGR to CHW-RGB
img = torch.from_numpy(img).float().unsqueeze(0)  # CHW-RGB to NCHW-RGB

with torch.no_grad():
    # pad input image to be a multiple of window_size
    window_size = 8
    _, _, h_old, w_old = img.size()
    h_pad = (h_old // window_size + 1) * window_size - h_old
    w_pad = (w_old // window_size + 1) * window_size - w_old
    img_lq = torch.cat([img, torch.flip(img, [2])], 2)[:, :, :h_old + h_pad, :]
    img_lq = torch.cat([img, torch.flip(img, [3])], 3)[:, :, :, :w_old + w_pad]
    # output = test(img, model, args, window_size)
    output = model(img)
    output = output[..., :h_old * 4, :w_old * 4]

# save image
output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
if output.ndim == 3:
    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))  # CHW-RGB to HCW-BGR
output = (output * 255.0).round().astype(np.uint8)  # float32 to uint8
cv2.imwrite(f'../images/output/output-SwinIR.png', output)
