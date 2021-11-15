from fisheye_model import fisheye_model
import cv2
import numpy as np
import torch
from torch import nn
import math
import os

# 相机内参
u = (256 - 1) / 2
v = (256 - 1) / 2
f = 1
fx = 128
fy = 128

K = np.array([
    [128, 0., u],
    [0., 128, v],
    [0., 0, f]
])


def get_map(k1, k2, k3, w, h):
    map1 = []
    map2 = []
    for x in range(w):
        for y in range(h):
            x0 = (x - u) / fx
            y0 = (y - v) / fy
            r2 = x0 * x0 + y0 * y0
            rp = np.sqrt(r2)
            t = np.arctan(rp / f)
            theta_d = t * (1 + k1 * t ** 2 + k2 * t ** 4 + k3 * t ** 6)
            rd = f * np.tan(theta_d)
            fi = np.arctan((y - v) / (x - u))
            xc = rd * np.cos(fi)
            yc = rd * np.sin(fi)
            x2 = xc * fx + u
            y2 = yc * fy + v
            map1.append(x2)
            map2.append(y2)
    map1 = np.array(map1).reshape(h, w).astype(np.float32)
    map2 = np.array(map2).reshape(h, w).astype(np.float32)
    return map1, map2


def get_img_path(root):
    res = []
    for d, folder, file in os.walk(root):
        for path in file:
            if path.split('.')[-1] == 'jpg':
                res.append(os.path.join(d, path))
    return res


# 参考论文：http://www.mdpi.com/1424-8220/16/6/807/pdf
def get_undistort_coefficient(D):
    res = []
    for d in D:
        k1, k2, k3 = d
        c1 = 0 - k1
        c2 = 3 * k1 * k1 - k2
        c3 = 8 * k1 * k2 - 12 * k1 * k1 * k1 - k3
        res.append([c1, c2, c3])
    return res


def apply_distort(img, d, w, h):
    k1, k2, k3 = d
    map1, map2 = get_map(k1, k2, k3, w, h)
    r_c = math.sqrt((map1[0][0]) ** 2 + (map2[0][0]) ** 2)
    distorted_img1 = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    img_r = cv2.rotate(img.copy(), cv2.ROTATE_180)
    distorted_img2 = cv2.remap(img_r, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    distorted_img = np.vstack((distorted_img1[:w // 2, :, :], distorted_img2[w // 2:, :, :]))
    distorted_img = cv2.rotate(distorted_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    distorted_img = cv2.flip(distorted_img, 1)
    return distorted_img


def ssim(img1, img2):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


class FishEyeModel(nn.Module):
    def __init__(self):
        super(FishEyeModel, self).__init__()
        self.model = fisheye_model(pretrained=False)

    def forward(self, x):
        """Forward"""
        x = self.model(x)
        return x


if __name__ == "__main__":
    img_path = "fish2.png"
    model = FishEyeModel().cuda()
    model.load_state_dict(torch.load(
        "D:/Design/Graduation/lightning_logs/version_7/checkpoints/epoch=6-step=34999.ckpt"
    )["state_dict"])
    model.eval()

    with torch.no_grad():
        img = cv2.imread(img_path)
        img = cv2.resize(img, (256, 256))
        distorted_img = img.copy()

        img = img.transpose(2, 0, 1)
        # to tensor
        img = torch.tensor(img).float().cuda()
        # normalize
        img = (img - 128) / 128

        pred = model(img.unsqueeze(0)).cpu()

        pred = pred.numpy().reshape(-1) / 10

        pred = pred.tolist()

        undistorted_img = apply_distort(distorted_img, pred, 256, 256)

        out_img = np.hstack([distorted_img, undistorted_img])
        cv2.imwrite("out2.jpg", out_img)
