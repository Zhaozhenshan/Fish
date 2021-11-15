from torch.utils.data import Dataset
import torch
import os
import numpy as np
from imgaug import augmenters as iaa
import cv2
import logging

#返回路径集合且支持随机存取
class FishEyeDataset(Dataset):
    # 初始化
    def __init__(self, root, mode):
        if mode not in ["train", "val", "test"]:
            raise ValueError("No such mode!")

        self.root = root
        self.mode = mode

        logging.info("Prepare for {mode} dataset...")
        self.prepare_data()
        logging.info("Ready")

    # 读数据
    def read_data(self):
        #原图路径列表
        img_paths = os.listdir(os.path.join(self.root, self.mode, "distort"))  # 返回文件或文件夹列表
        img_paths = sorted(img_paths, key=lambda x: int(x.split(".")[0]))  # 以.为分隔符，返回分割后列表
        imgs = {}
        for i in range(len(img_paths)):
            imgs[i] = os.path.join(self.root, self.mode, "distort", img_paths[i])  # 路径拼接函数

        # 原图路径列表
        ori_imgs = {}
        if self.mode == "test":
            ori_img_paths = os.listdir(os.path.join(self.root, self.mode, "origin"))
            ori_img_paths = sorted(ori_img_paths, key=lambda x: int(x.split(".")[0]))
            for i in range(len(ori_img_paths)):
                ori_imgs[i] = os.path.join(self.root, self.mode, "origin", ori_img_paths[i])

        # 畸变图标签
        labels = {}
        with open(os.path.join(self.root, self.mode, 'label.txt'), 'r') as f:
            data = f.read().strip().split("\n") #读取每行数据
            for line in data:
                line = line.strip().split(",")  #每行三个数据
                labels[int(line[0])] = [float(e) for e in line[1:]]
        #原图路径列表 畸变图路径列表 标签值
        return imgs, ori_imgs, labels

    #准备数据函数
    def prepare_data(self):
        self.imgs, self.ori_imgs, self.labels = self.read_data()

    #imgaug库用于图像增强
    def img_augment(self, img):
        aug_seq = iaa.Sequential([
            # 模糊处理或者锐化处理
            iaa.OneOf([
                iaa.GaussianBlur((0, 1)),
                # iaa.MedianBlur(k=(3,5)),
                iaa.Sharpen(alpha=(0, 0.3)),
            ]),
            # 改变亮度
            iaa.Multiply((0.7, 1.2), per_channel=0.5),
            # 增加对比度
            iaa.LinearContrast((0.7, 1.2), per_channel=0.5),
        ], random_order=True)

        return aug_seq(image=img)

    #获取图片的一维长度
    def __len__(self):
        return len(self.imgs)

    #根据路径获取图片
    def __getitem__(self, index):
        img = cv2.imread(self.imgs[index])  #原图
        label = self.labels[index][3:]      #校正参数 c1 c2 c3
        if self.mode == 'train':
            # 图像增强
            img = self.img_augment(img)

            # 矩阵转置
            img = img.transpose(2, 0, 1)
            img = torch.tensor(img).float()

            #(0-1)标准化
            img = (img - 128) / 128
            return img, np.array(label).astype(np.float32) * 10
        elif self.mode == "val":
            # transpose
            img = img.transpose(2, 0, 1)
            img = torch.tensor(img).float()
            # normalize
            img = (img - 128) / 128
            return img, np.array(label).astype(np.float32) * 10
        else:
            img = img.transpose(2, 0, 1)
            img = torch.tensor(img).float()
            # normalize
            img = (img - 128) / 128
            ori_img = cv2.imread(self.ori_imgs[index])
            return img, ori_img, np.array(label).astype(np.float32) * 10


# 当模块是被导入模块是，以下内容不执行
if __name__ == "__main__":
    from torch.utils.data import DataLoader

    dataset = FishEyeDataset("data", "train")
    # print(dataset.imgs)
    dataloader = DataLoader(dataset=dataset, batch_size=8, num_workers=0, shuffle=True)
    for batch_idx, (imgs, label) in enumerate(dataloader):
        imgs = imgs.numpy()
        print(imgs.shape)
