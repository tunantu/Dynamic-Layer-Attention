import torch
import torch.nn as nn
import torchvision.models as models
import copy
__all__ = ['senet_50']

# 自定义SEBlock
# 自定义SEBlock
class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        se_weights = self.se(x)
        return x * se_weights

# 创建自定义Bottleneck类
class CustomBottleneck(models.resnet.Bottleneck):
    def __init__(self, *args, **kwargs):
        super(CustomBottleneck, self).__init__(*args, **kwargs)
        self.out_channels = self.conv3.out_channels
        self.se = SEBlock(self.out_channels)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out

# 创建自定义ResNet-50模型
class SE_ResNet50(models.ResNet):
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False, groups=1, width_per_group=64, replace_stride_with_dilation=None, norm_layer=None):
        super(SE_ResNet50, self).__init__(block, layers, num_classes, zero_init_residual, groups, width_per_group, replace_stride_with_dilation, norm_layer)

# 创建自定义ResNet-50模型，并加载预训练参数
se_resnet = SE_ResNet50(CustomBottleneck, [3, 4, 6, 3])

# 加载预训练参数，忽略SEBlock的权重
pretrained_resnet = models.resnet50(pretrained=True)
state_dict = pretrained_resnet.state_dict()

# 移除SEBlock的权重
state_dict = {k: v for k, v in state_dict.items() if not k.startswith('layer') or not k.startswith('se')}

# 加载剩余的预训练参数
se_resnet.load_state_dict(state_dict, strict=False)

# 修改全连接层的输出类别数
# se_resnet.fc = nn.Linear(2048, 1000)  # 1000是自定义输出的类别数

def senet_50(**kwargs):
    return se_resnet

