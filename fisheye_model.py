import torch
import torch.nn as nn

#反卷积操作          (256 28 28)   -->   (32 224 224)
class Deconv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Deconv, self).__init__()
        self.d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        # self.b = nn.BatchNorm2d(out_channels)
        self.r = nn.ReLU()

    def forward(self, x):
        b, c, h, w = x.size()
        x = self.d(x, output_size=torch.Size((b, c, h * 2, w * 2)))
        # x = self.b(x)
        x = self.r(x)
        return x


class FishEyeModel(nn.Module):

    def __init__(self, features, num_classes=3, init_weights=True):
        super(FishEyeModel, self).__init__()
        self.features = features
        print(self.features)
        #反卷积层
        self.deconv1 = Deconv(256, 128)      #初试图像为256*28*28
        self.deconv2 = Deconv(128, 64)
        self.deconv3 = Deconv(64, 32)        #反卷积操作到32*244*244

        #卷积层
        self.undistort_net = nn.Sequential(
            self.cbr(96, 64), self.cbr(64, 64),
            nn.MaxPool2d(2, 2),
            self.cbr(64, 128), self.cbr(128, 128),
            nn.MaxPool2d(2, 2),
            self.cbr(128, 256), self.cbr(256, 256),
            nn.MaxPool2d(2, 2),
            self.cbr(256, 256), self.cbr(256, 256),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.7)
        )
        #全连接层
        self.fc1 = nn.Linear(256 * 16 * 16, 256)
        self.drop = nn.Dropout(0.7)
        self.fc2 = nn.Linear(256, 3)
        self.relu = nn.ReLU()

        if init_weights:
            self._initialize_weights()

    def cbr(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            # nn.BatchNorm2d(out_c), 
            nn.ReLU(inplace=True)
        )

    #前向传播 features
    def forward(self, x):
        x1 = self.features[:4](x)              #取第一层的参数

        x2 = self.features[4:17](x1)           #取最大池化层的参数

        x3 = self.deconv1(x2)
        x3 = self.deconv2(x3)
        x3 = self.deconv3(x3)
        x_fusion = torch.cat([x1, x3], dim=1)  #将两个张量拼接在一起

        x_out = self.undistort_net(x_fusion)   #卷积层

        x_out = torch.flatten(x_out, 1)        #一维数组

        x_out = self.fc1(x_out)                #全连接层:第一层
        x_out = self.relu(x_out)
        x_out = self.drop(x_out)

        x_out = self.fc2(x_out)                #全连接层:第二层
        return x_out

    #初始化权重
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


def _fisheye(cfg, batch_norm, pretrained=True, pretrain_path=".", **kwargs):
    if pretrained:
        kwargs['init_weights'] = False
    model = FishEyeModel(make_layers(cfg, batch_norm=batch_norm), **kwargs)
    if pretrained:
        state_dict = torch.load(pretrain_path)
        model.load_state_dict(state_dict, strict=False)
    return model


cfg = [
    64, 64, 'M',
    128, 128, 'M',
    256, 256, 256, 'M',
    512, 512, 512, 'M',
    512, 512, 512, 'M'
]


def fisheye_model(with_batch_norm=False, pretrained=False, pretrain_path="pretrained/vgg16_bn-6c64b313.pth", **kwargs):
    return _fisheye(cfg, with_batch_norm, pretrained, pretrain_path, **kwargs)


if __name__ == "__main__":
    x = torch.randn(1, 3, 256, 256)
    model = fisheye_model()
    y = model(x)
    print(y.size())
# print(model)
