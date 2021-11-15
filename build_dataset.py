import os
import random
import cv2
import numpy as np
import math

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


def get_distort_coefficient(path):
    res = []
    with open(path, 'r') as f:
        data = f.read().strip().split("\n")
        for line in data:
            res.append([float(e) for e in line.split(",")])
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


def distort_imgs(img_paths, out_root, distort_c, undistort_c, size):
    os.makedirs(os.path.join(out_root, 'origin'), exist_ok=True)
    os.makedirs(os.path.join(out_root, 'distort'), exist_ok=True)
    count = 0
    label_res = []

    for path in img_paths:
        img = cv2.imread(path)
        img = cv2.resize(img, size, interpolation=cv2.INTER_CUBIC)
        w, h = size
        for idx, d_c in enumerate(distort_c):
            distorted_img = apply_distort(img, d_c, w, h)
            cv2.imwrite(os.path.join(out_root, 'distort', str(count) + ".jpg"), distorted_img)
            cv2.imwrite(os.path.join(out_root, 'origin', str(count) + ".jpg"), img)
            label_res.append(str(count) + "," + ",".join([str(e) for e in d_c + undistort_c[idx]]))
            count += 1

    with open(os.path.join(out_root, 'label.txt'), 'w') as f:
        f.write("\n".join(label_res))


def build(img_root, out_root, distort_file, train_num, val_num, test_num, size):
    img_paths = get_img_path(img_root)

    random.shuffle(img_paths)

    img_paths = img_paths[:(train_num + val_num + test_num)]

    distort_c = get_distort_coefficient(distort_file)
    undistort_c = get_undistort_coefficient(distort_c)

    os.makedirs(out_root, exist_ok=True)
    os.makedirs(os.path.join(out_root, 'train'), exist_ok=True)
    os.makedirs(os.path.join(out_root, 'val'), exist_ok=True)
    os.makedirs(os.path.join(out_root, 'test'), exist_ok=True)

    distort_imgs(
        img_paths[:train_num],
        os.path.join(out_root, 'train'),
        distort_c,
        undistort_c,
        size
    )

    distort_imgs(
        img_paths[train_num:train_num + val_num],
        os.path.join(out_root, 'val'),
        distort_c,
        undistort_c,
        size
    )

    distort_imgs(
        img_paths[train_num + val_num:],
        os.path.join(out_root, 'test'),
        distort_c,
        undistort_c,
        size
    )


if __name__ == '__main__':
    build(
        img_root='ADE20K_2016_07_26',  # ADE20数据集目录
        out_root='data',  # 输出数据集目录
        distort_file='distort_coefficient.txt',  # 畸变参数文件
        train_num=2000,  # 挑选训练图片数
        val_num=100,  # 挑选验证图片数
        test_num=100,  # 挑选测试图片数
        size=(256, 256)  # 目标图片尺寸
    )
