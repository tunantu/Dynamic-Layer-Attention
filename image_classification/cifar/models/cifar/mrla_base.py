import torch
from torch import nn
from torch.nn.parameter import Parameter
from math import sqrt
from math import log
from einops import rearrange
import math
import torch.nn.functional as F

__all__ = ['mrla_base']

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class mrla_base_layer(nn.Module):
    def __init__(self, input_dim, heads=None, dim_perhead=None, k_size=None, init_cell=False):
        super(mrla_base_layer, self).__init__()
        self.input_dim = input_dim
        self.init_cell = init_cell
        
        if (heads == None) and (dim_perhead == None):
            raise ValueError("arguments heads and dim_perhead cannot be None at the same time !")
        elif dim_perhead != None:
            heads = int(input_dim / dim_perhead)
        else:
            heads = heads
        self.heads = heads
        
        if k_size == None:
            t = int(abs((log(input_dim, 2) + 1) / 2.))
            k_size = t if t % 2 else t+1
        self.k_size = k_size
    
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.Wq = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.Wk = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.Wv = nn.Conv2d(input_dim, input_dim, kernel_size=3, stride=1, padding=1, groups=input_dim, bias=False) 
        self._norm_fact = 1 / sqrt(input_dim / heads)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, prev_K, prev_V):
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size()
        # feature descriptor on the global spatial information
        y = self.avg_pool(x) # [b, c, 1, 1]
        y = y.squeeze(-1).transpose(-1, -2) # [b, 1, c]
        Q = self.Wq(y) # Q: [b, 1, c] 
        k = self.Wk(y) # k: [b, 1, c]
        v = self.Wv(x) # v: [b, c, h, w]
        
        if self.init_cell:
            K = k # K: [b, 1, c]
            V = v.unsqueeze(1) # V: [b, 1, c, h, w]
            
        else:        
            K = torch.cat((prev_K, k), dim=1) # K: [b, t, c]
            V = torch.cat((prev_V, v.unsqueeze(1)), dim=1) # V: [b, t, c, h, w]
        output_K = K
        output_V = V
      
        Q = Q.view(b, self.heads, 1, int(c/self.heads)) # [b, g, 1, c/g]
        K = rearrange(K, 'b t (g d) -> b g t d', b=b, g=self.heads, d=int(c/self.heads)) # [b, g, t, c/g]
        V = rearrange(V, 'b t (g d) h w -> b g t d h w', g=self.heads, d=int(c/self.heads)) # [b, g, t, c/g, h, w]
        atten = torch.einsum('... i d, ... j d -> ... i j', Q, K) * self._norm_fact
        atten = self.softmax(atten)
        V = rearrange(V, 'b g t d h w -> b g t (d h w)')
        # output = atten @ V # [b g 1 (c/g h w)]
        output = torch.einsum('bgit, bgtj -> bgij', atten, V) 
        output = output.unsqueeze(2).reshape(b, c, h, w)
        return output, output_K, output_V

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class mrla_module(nn.Module):
    dim_perhead = 32  
    def __init__(self, input_dim, init_cell=False, channel_wise=False):
        super(mrla_module, self).__init__()
        if channel_wise:
            self.dim_perhead = 1
        self.mrla = mrla_base_layer(input_dim=input_dim, dim_perhead=self.dim_perhead, init_cell=init_cell) 
        self.init_cell = init_cell
        
    def forward(self, xt, prev_k, prev_v):
        if self.init_cell: # 1st layer in each stage
           prev_k = None
           prev_v = None 
        out, kt, vt = self.mrla(xt, prev_k, prev_v)
        return out, kt, vt



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, init_cell=False, drop_path = 0.0):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.mrla = mrla_module(input_dim= planes*self.expansion,init_cell=init_cell)
        self.bn_mrla = nn.BatchNorm2d(planes * self.expansion)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
    def forward(self, x, prev_k, prev_v):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(residual)
        out = out + residual
        out = self.relu(out)
        attn_t, k, v = self.mrla(out, prev_k, prev_v)
        attn_t = self.bn_mrla(attn_t)
        attn_t = self.relu(attn_t)
        out = out + self.drop_path(attn_t)

        return out, k, v



class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, init_cell=False, drop_path = 0.0):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.mrla = mrla_module(input_dim= planes*self.expansion,init_cell=init_cell)
        self.bn_mrla = nn.BatchNorm2d(planes * self.expansion)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
    def forward(self, x, prev_k, prev_v):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(residual)

        out += residual
        out = self.relu(out)
        attn_t, k, v = self.mrla(out, prev_k, prev_v)
        attn_t = self.bn_mrla(attn_t)
        attn_t = self.relu(attn_t)
        out = out + self.drop_path(attn_t)
        return out, k, v


class ResNet(nn.Module):

    def __init__(self, depth, num_classes=1000, block_name='BasicBlock', drop_path =0.0):
        super(ResNet, self).__init__()
        # Model type specifies number of layers for CIFAR-10 model
        if block_name.lower() == 'basicblock':
            assert (depth - 2) % 6 == 0, 'When use basicblock, depth should be 6n+2, e.g. 20, 32, 44, 56, 110, 1202'
            n = (depth - 2) // 6
            block = BasicBlock
        elif block_name.lower() == 'bottleneck':
            assert (depth - 2) % 9 == 0, 'When use bottleneck, depth should be 9n+2, e.g. 20, 29, 47, 56, 110, 1199'
            n = (depth - 2) // 9
            block = Bottleneck
        else:
            raise ValueError('block_name shoule be Basicblock or Bottleneck')


        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        
        stages = [None]*3
        stages[0] = self._make_layer(block, 16, n, stride=1,init_cell = True, drop_path=drop_path)
        stages[1] = self._make_layer(block, 32, n, stride=2,init_cell=True, drop_path=drop_path)
        stages[2] = self._make_layer(block, 64, n, stride=2,init_cell=True, drop_path=drop_path)
        self.stages = nn.ModuleList(stages)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, init_cell=False, drop_path=0.0):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, init_cell, drop_path = drop_path))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes,planes, drop_path = drop_path))
        return nn.ModuleList(layers)

    def forward_features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        k = None
        v = None
        for layers in self.stages:
            for layer in layers:
                x, k, v = layer(x, k, v)
        return x
    def forward(self, x):
        x = self.forward_features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def mrla_base(**kwargs):
    """
    Constructs a ResNet model.
    """
    return ResNet(**kwargs)
