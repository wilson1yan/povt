from ctypes import resize
import math
import numpy as np
import wandb

import torch
import torch.nn.functional as F


class Quantizer(object):
    def __init__(self, min, max, n_values):
        self.min = min
        self.max = max
        self.n_values = n_values

    def quantize(self, x):
        x = (x - self.min) / (self.max - self.min)
        x = x * self.n_values
        x = x.floor().clamp(0, self.n_values - 1).long()
        return x
    
    def dequantize(self, x):
        x = x.float() / (self.n_values - 1)
        x = x * (self.max - self.min) + self.min
        return x

        
def resize_bbox(bbox, shift, scale):
    x1, y1, x2, y2 = bbox
    shift_h, shift_w = shift
    scale_h, scale_w = scale

    x1, x2 = x1 * scale_w, x2 * scale_w
    y1, y2 = y1 * scale_h, y2 * scale_w

    x1, x2 = x1 - shift_w, x2 - shift_w
    y1, y2 = y1 - shift_h, y2 - shift_h

    bbox = np.array([x1, y1, x2, y2])
    return bbox

# reshapes tensor start from dim i (inclusive)
# to dim j (exclusive) to the desired shape
# e.g. if x.shape = (b, thw, c) then
# view_range(x, 1, 2, (t, h, w)) returns
# x of shape (b, t, h, w, c)
def view_range(x, i, j, shape):
    shape = tuple(shape)

    n_dims = len(x.shape)
    if i < 0:
        i = n_dims + i

    if j is None:
        j = n_dims
    elif j < 0:
        j = n_dims + j

    assert 0 <= i < j <= n_dims

    x_shape = x.shape
    target_shape = x_shape[:i] + shape + x_shape[j:]
    return x.view(target_shape)

    
def tensor_slice(x, begin, size):
    assert all([b >= 0 for b in begin])
    size = [l - b if s == -1 else s
            for s, b, l in zip(size, begin, x.shape)]
    assert all([s >= 0 for s in size])

    slices = [slice(b, b + s) for b, s in zip(begin, size)]
    return x[slices]

    
def masked_select(tensor, mask):
    # tensor: [d_1, d_2, ..., d_n, *], mask: [d_1, d_2, ..., d_n]
    # returns: M*, with tensors selected at only valid (unmasked) positions
    shape = mask.shape
    n_dim = len(shape)
    assert len(tensor.shape) >= n_dim

    idxs = torch.nonzero(mask, as_tuple=False) # M(n_dim)
    block_sizes = np.concatenate(([1], np.cumprod(shape[::-1])[:-1]))[::-1]
    block_sizes = torch.LongTensor(block_sizes.copy()).to(tensor.device) # n_dim
    idxs = torch.sum(idxs * block_sizes.unsqueeze(0), dim=1) # M

    tensor = tensor.flatten(end_dim=n_dim - 1) # (d_1d_2...d_n)*
    tensor = torch.index_select(tensor, 0, idxs) # M*
    return tensor


def inverse_masked_select(tensor, mask):
    # tensor: M*, mask: [d_1, d_2, ..., d_n]
    # returns: [d_1, d_2, ..., d_n, *] with zeros in the invalid (masked) positions
    shape = mask.shape
    n_dim = len(shape)
    out = torch.zeros(*shape, *tensor.shape[1:], 
                      device=tensor.device, dtype=tensor.dtype)
    out = out.flatten(end_dim=n_dim - 1) # (d_1d_2...d_n)*
    
    idxs = torch.nonzero(mask, as_tuple=False) # M(n_dim)
    block_sizes = np.concatenate(([1], np.cumprod(shape[::-1])[:-1]))[::-1]
    block_sizes = torch.LongTensor(block_sizes.copy()).to(tensor.device) # n_dim
    idxs = torch.sum(idxs * block_sizes.unsqueeze(0), dim=1) # M

    out = torch.index_add(out, 0, idxs, tensor)
    out = view_range(out, 0, 1, shape)
    return out

    
def extract_glimpses(images, bboxes, valid_bboxes, glimpse_size,
                     return_padded=True, return_bboxes=False):
    # images: BCHW, bboxes: BK4, valid_bboxes: BK
    images_orig = images
    images = images.float()
    if isinstance(glimpse_size, int):
        glimpse_size = (glimpse_size, glimpse_size)
    n_valid = valid_bboxes.sum(dim=1) # B

    images = images.repeat_interleave(n_valid, dim=0) # MCHW

    bboxes = masked_select(bboxes, valid_bboxes) # M4
    assert bboxes.shape[0] == images.shape[0]

    if bboxes.shape[0] == 0:
        out = torch.zeros((0, images.shape[1], *glimpse_size),
                          device=images.device, dtype=images.dtype)
    else:
        shift, scale = bboxes.chunk(2, dim=1) # M2, M2
        out = spatial_transform(images, shift, scale, 
                                (images.shape[0], images.shape[1], 
                                *glimpse_size)) # MCGG
    if return_padded:
        out = inverse_masked_select(out, valid_bboxes) # BKCGG
    out = out.type_as(images_orig)
    
    if return_bboxes:
        return out, bboxes
    return out


def bbox_image_to_norm(bboxes, H, W):
    # bbox: *4
    # shift lies in [-1, 1]
    shift_y = (bboxes[..., 0] + bboxes[..., 2]) / 2
    shift_y = (shift_y - H / 2) / (H / 2)
    shift_x = (bboxes[..., 1] + bboxes[..., 3]) / 2
    shift_x = (shift_x - W / 2) / (W / 2)
    shift = torch.stack((shift_x, shift_y), dim=-1) # *2

    # scale lies in [0, 1]
    height = (bboxes[..., 2] - bboxes[..., 0]) / H
    width = (bboxes[..., 3] - bboxes[..., 1]) / W
    scale = torch.stack((width, height), dim=-1) # *2

    return torch.cat((shift, scale), dim=-1) # *4


def bbox_norm_to_image(bbox, H, W):
    # bbox: *4
    bbox_shift, bbox_scale = bbox.chunk(2, dim=-1) # *2 (x2)

    width = bbox_scale[..., 0] * W
    height = bbox_scale[..., 1] * H

    center_r = bbox_shift[..., 1] * (H / 2) + (H / 2)
    center_c = bbox_shift[..., 0] * (W / 2) + (W / 2)
    tl_r = torch.round(center_r - height / 2)
    tl_c = torch.round(center_c - width / 2)
    br_r = torch.round(center_r + height / 2)
    br_c = torch.round(center_c + width / 2)

    tl_r = torch.clamp(tl_r, 0, H - 1).long() # *
    tl_c = torch.clamp(tl_c, 0, W - 1).long()
    br_r = torch.clamp(br_r, 0, H).long()
    br_c = torch.clamp(br_c, 0, W).long()

    return torch.stack((tl_r, tl_c, br_r, br_c), dim=-1) # *4

def flatten_idx(idx, shape):
    assert len(idx) == len(shape)
    shape = shape[1:][::-1]
    cum = np.concatenate(([1], np.cumprod(shape)))[::-1]
    idx = np.array(idx)
    flat_idx = np.sum(idx * cum)
    return flat_idx


def quantize_bboxes(bboxes, shift_quantizer=None,
                    scale_quantizer=None):
    if shift_quantizer is None:
        shift_quantizer = Quantizer(-1, 1, 64)
        scale_quantizer = Quantizer(0, 1, 64)

    shift, scale = bboxes.chunk(2, dim=-1) # BK2 (x2)
    shift = shift_quantizer.quantize(shift)
    scale = scale_quantizer.quantize(scale)
    bboxes_quantized = torch.cat([shift, scale], dim=-1) # BK4
    
    shift = shift_quantizer.dequantize(shift)
    scale = scale_quantizer.dequantize(scale)
    bboxes = torch.cat([shift, scale], dim=-1)
    return bboxes, bboxes_quantized


def spatial_transform(obs, shift, scale, out_shape, inverse=False):
    # obs:  BCHW
    # shift, scale: B2
    theta = torch.zeros(2, 3).repeat(obs.shape[0], 1, 1).to(obs.device)
    theta[:, 0, 0] = scale[:, 0] if not inverse else 1 / (scale[:, 0] + 1e-9)
    theta[:, 1, 1] = scale[:, 1] if not inverse else 1 / (scale[:, 1] + 1e-9)
    theta[:, :, -1] = shift if not inverse else -shift / (scale + 1e-9)

    grid = F.affine_grid(theta, out_shape).float()
    return F.grid_sample(obs, grid)


import math
import numpy as np
from moviepy.editor import ImageSequenceClip
def save_video_grid(video, fname=None, nrow=None):
    b, t, c, h, w = video.shape
    video = video.movedim(2, -1)
    video = (video.cpu().numpy() * 255).astype('uint8')

    if nrow is None:
        nrow = math.ceil(math.sqrt(b))
    ncol = math.ceil(b / nrow)
    padding = 1
    video_grid = np.zeros((t, (padding + h) * ncol + padding,
                           (padding + w) * nrow + padding, c), dtype='uint8')
    for i in range(b):
        r = i // nrow
        c = i % nrow

        start_r = (padding + h) * r
        start_c = (padding + w) * c
        video_grid[:, start_r:start_r + h, start_c:start_c + w] = video[i]


    if fname is not None:
        clip = ImageSequenceClip(list(video_grid), fps=5)
        clip.write_gif(fname, fps=5)
        print('saved videos to', fname)

    return video_grid # THWC, uint8


# Metrics
def compute_metric(prediction, ground_truth, metric_fn):
    # BCTHW
    assert prediction.shape == ground_truth.shape
    B, T = prediction.shape[0], prediction.shape[2]
    prediction = prediction.movedim(2, 1).flatten(end_dim=1) # (BT)CHW
    ground_truth = ground_truth.movedim(2, 1).flatten(end_dim=1)

    metrics = metric_fn(prediction, ground_truth) # (BT)
    metrics = metrics.view(B, T) # BT

    metric_per_timestep = metrics.mean(dim=0)

    return metric_per_timestep


# all methods below take as input pairs of images
# of shape BCHW. They DO NOT reduce batch dimension
# NOTE: Assumes that images are in [0, 1]

import lpips
def get_lpips(device):
    lpips_module = lpips.LPIPS().to(device)
    def fn(imgs1, imgs2):
        with torch.no_grad():
            imgs1, imgs2 = 2 * imgs1 - 1, 2 * imgs2 - 1
            lpips_metric = lpips_module(imgs1, imgs2)
            lpips_metric = lpips_metric.flatten(start_dim=1).squeeze(1)
            return lpips_metric
    return lambda imgs1, imgs2: compute_metric(imgs1, imgs2, fn)

def get_ssim(device):
    ssim_module = SSIM(window_size=5).to(device)
    def fn(imgs1, imgs2):
        ssim_metric = ssim_module(imgs1, imgs2)
        return ssim_metric
    return lambda imgs1, imgs2: compute_metric(imgs1, imgs2, fn)

def get_psnr():
    return lambda imgs1, imgs2: compute_metric(imgs1, imgs2, psnr_metric)

def psnr_metric(prediction, ground_truth):
    mse = F.mse_loss(prediction, ground_truth, reduction='none')
    mse = mse.flatten(start_dim=1).mean(dim=1)
    psnr = 10 * torch.log10(1. / mse)
    return psnr

# SSIM
def gaussian(window_size, sigma):
    gauss = torch.FloatTensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) 
                               for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIM(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = False):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 3
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel


        return _ssim(img1, img2, window, self.window_size, channel, self.size_average) 

    
if __name__ == '__main__':
    import torch
    from torchvision import transforms
    from PIL import Image
    import numpy as np
    device = torch.device('cuda')

    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor()
    ])

    img = Image.open('fox.jpg') # HWC
    img = transform(img)

    imgs1 = img.unsqueeze(0).unsqueeze(2)
    imgs1 = imgs1.repeat(4, 1, 2, 1, 1).to(device)
    imgs2 = (imgs1 + torch.randn_like(imgs1) * 0.01).clamp(0, 1)

    lpips_fn = get_lpips(device)
    ssim_fn = get_ssim(device)
    psnr_fn = get_psnr()

    print(psnr_fn(imgs1, imgs2))
    print(ssim_fn(imgs1, imgs2))
    print(lpips_fn(imgs1, imgs2))
