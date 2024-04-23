import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules.light_layer_attention import light_layer_attention
from models.modules.eca_module import eca_layer
from models.modules.se_module import se_layer
from models.utils.drop import DropPath
from math import log

__all__ = [
           'resnet50_dla_l', 
           'resnet101_dla_l', 
           ]

class dsu_cell(nn.Module):
    def __init__(self, input_size, hidden_size):
        """"Constructor of the class"""
        super(dsu_cell, self).__init__()
        self.seq = nn.Sequential(nn.Linear(input_size, input_size // 16),
                      nn.ReLU(inplace=True),
                      nn.Linear(input_size // 16, 3 * hidden_size))
    def forward(self,x):
        return self.seq(x)

class DSU_block(nn.Module):
    def __init__(self, input_size, hidden_size, nlayers, dropout = 0.1):
        """"Constructor of the class"""
        super(DSU_block, self).__init__()

        self.nlayers = nlayers
        self.dropout = nn.Dropout(p=dropout)

        ih, hh = [], []
        for i in range(nlayers):
            if i==0:
                ih.append(dsu_cell(input_size, hidden_size))
                hh.append(dsu_cell(hidden_size, hidden_size))
            else:
                ih.append(nn.Linear(hidden_size, 3 * hidden_size))
                hh.append(nn.Linear(hidden_size, 3 * hidden_size))
        self.w_ih = nn.ModuleList(ih)
        self.w_hh = nn.ModuleList(hh)
        self.sigmoid = nn.Sigmoid()
    def forward(self, input, hidden):
        """"Defines the forward computation of the DSU_block"""
        hy, cy = [], []
        for i in range(self.nlayers):
            hx, cx = hidden[0][i], hidden[1][i]
            gates = self.w_ih[i](input) + self.w_hh[i](hx)
            i_gate, f_gate, c_gate = gates.chunk(3, 1)
            i_gate = torch.sigmoid(i_gate)
            f_gate = torch.sigmoid(f_gate)
            c_gate = torch.tanh(c_gate)
            ncx = (f_gate * cx) + (i_gate * c_gate)
            nhx = torch.sigmoid(ncx)
            cy.append(ncx)
            hy.append(nhx)
            input = self.dropout(nhx)

        hy, cy = torch.stack(hy, 0), torch.stack(cy, 0) 
        return hy, cy



def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)
    
def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class light_layer_attention_module(nn.Module):
    dim_perhead = 32
    
    def __init__(self, input_dim):
        super(light_layer_attention_module, self).__init__()
        self.mrla = light_layer_attention(input_dim=input_dim, dim_perhead=self.dim_perhead)
        self.lambda_t = nn.Parameter(torch.randn(input_dim, 1, 1))  

        
    def forward(self, xt, ot_1):
        attn_t = self.mrla(xt)
        out = attn_t + self.lambda_t.expand_as(ot_1) * ot_1 
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, 
                 stride=1, downsample=None, 
                 SE=False, ECA_size=None, 
                 groups=1, base_width=64, dilation=1, 
                 norm_layer=nn.BatchNorm2d, drop_path=0.0):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
            # norm_layer = nn.SyncBatchNorm 
            # # support for multi gpus
            # model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
            
        width = int(planes * (base_width / 64.)) * groups
        
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        
        self.downsample = downsample
        self.stride = stride
        
        # channel attention modules
        self.se = None
        if SE:
            self.se = se_layer(planes * self.expansion, reduction=16)
        self.eca = None
        if ECA_size != None:
            self.eca = eca_layer(planes * self.expansion, int(ECA_size))
            

        
    def forward(self, x):
        identity = x
        
        # res block
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        
        # channel attention
        if self.se != None:
            out = self.se(out)
        if self.eca != None:
            out = self.eca(out) 
        # downsampling for short cut    
        if self.downsample is not None:
            identity = self.downsample(identity)
            
        out += identity 
        out = self.relu(out)        
        return out, identity


class layer_attention(nn.Module):
    expansion = 4
    def __init__(self, planes, drop_path=0.0):
        super(layer_attention, self).__init__()
        self.layer_attention = light_layer_attention_module(input_dim= planes*self.expansion)
        self.bn_layer_attention = nn.BatchNorm2d(planes * self.expansion)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
    def forward(self, x, residual):
        out = x + self.drop_path(self.bn_layer_attention(self.layer_attention(x, residual)))
        return out

class Attention(nn.Module):
    def __init__(self, ModuleList, block_idx):
        super(Attention, self).__init__()
        self.layers, self.attention = ModuleList
        # self.back = back
        if block_idx == 1:
            self.lstm = DSU_block(256, 256, 1)
            input_dim = 256
        elif block_idx == 2:
            self.lstm = DSU_block(512, 512, 1)
            input_dim =512
        elif block_idx == 3:
            self.lstm = DSU_block(1024, 1024, 1)
            input_dim=1024
        elif block_idx == 4:
            self.lstm = DSU_block(2048, 2048, 1)
            input_dim=2048
            
        self.GlobalAvg = nn.AdaptiveAvgPool2d((1, 1))
        self.GlobalAvg1 = nn.AdaptiveAvgPool2d((1, 1))
        self.relu1 = nn.ReLU(inplace=True)
        self.block_idx = block_idx
        self.sigmoid = nn.Sigmoid()
        self.relu2 = nn.ReLU(inplace=True)
        self.extict = nn.Conv2d(input_dim, input_dim, kernel_size=3, stride=1, padding=1, groups=input_dim, bias=False)
        

    def forward(self, x):
        for idx, (layer, attention) in enumerate(zip(self.layers, self.attention)):

            x, org = layer(x)  

            if idx == 0:
                seq = self.GlobalAvg(x)
                list = seq.view(seq.size(0), 1, seq.size(1))
                seq = seq.view(seq.size(0), seq.size(1))
                ht = torch.zeros(1, seq.size(0), seq.size(1)).cuda()  # 1 mean number of layers
                ct = torch.zeros(1, seq.size(0), seq.size(1)).cuda()
                ht, ct = self.lstm(seq, (ht, ct))  # 1 * batch size * length
            else:
                seq = self.GlobalAvg(x)
                list = torch.cat((list, seq.view(seq.size(0), 1, seq.size(1))), 1)
                seq = seq.view(seq.size(0), seq.size(1))
                ht, ct = self.lstm(seq, (ht, ct))
            h_in = ht[-1]
            c_in = ct[-1]
            y_in = self.GlobalAvg1(org)
            y_in = y_in.view(y_in.size(0), y_in.size(1))
            
            out, _ = self.lstm(y_in, (h_in, c_in))
            out = out[-1].view(out.size(1), out.size(2), 1, 1)
            
            re = self.extict(org)
            org = org * out + re
            x = attention(x, org)
            
        return x

    

class ResNet(nn.Module):
    def __init__(self, block, 
                layers, 
                num_classes=1000, 
                SE=False, 
                ECA=None, 
                zero_init_last_bn=True, #zero_init_residual=False,
                groups=1, 
                width_per_group=64, 
                replace_stride_with_dilation=None,
                norm_layer=nn.BatchNorm2d, 
                drop_rate=0.0, 
                drop_path=0.0
                ):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
            # norm_layer = nn.SyncBatchNorm
        self._norm_layer = norm_layer
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        self.drop_path = drop_path
        self.layer_attention = layer_attention
        
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        
        if ECA is None:
            ECA = [None] * 4
        elif len(ECA) != 4:
            raise ValueError("argument ECA should be a 4-element tuple, got {}".format(ECA))
    
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = Attention( self._make_layer(block, 64, layers[0], SE=SE, ECA_size=ECA[0]),1)
        self.layer2 = Attention( self._make_layer(block, 128, layers[1], SE=SE, ECA_size=ECA[1], stride=2, dilate=replace_stride_with_dilation[0]),2)
        self.layer3 = Attention( self._make_layer(block, 256, layers[2], SE=SE, ECA_size=ECA[2], stride=2, dilate=replace_stride_with_dilation[1]),3)
        self.layer4 = Attention( self._make_layer(block, 512, layers[3], SE=SE, ECA_size=ECA[3], stride=2, dilate=replace_stride_with_dilation[2]),4)
        
        # classifier head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
            
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            # elif isinstance(m, (nn.SyncBatchNorm, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
        # Zero-initialize the last BN in each residual branch
        if zero_init_last_bn:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, block, planes, blocks, SE, ECA_size, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            ) # downsampling and change channels for x(identity)

        layers = []
        mrla=[]
        layers.append(block(self.inplanes, planes, stride, downsample, 
                            SE=SE, ECA_size=ECA_size, groups=self.groups,
                            base_width=self.base_width, dilation=previous_dilation, norm_layer=norm_layer, drop_path=self.drop_path))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, 
                                SE=SE, ECA_size=ECA_size, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, drop_path=self.drop_path))
        for i in range(0, blocks):
            mrla.append(self.layer_attention(planes, drop_path=self.drop_path))
        return nn.Sequential(*layers), nn.Sequential(*mrla)
        # return nn.ModuleList(layers)

    def forward_features(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
      
        return x
    
    def forward(self, x):
        x = self.forward_features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        if self.drop_rate:
            x = F.dropout(x, p=float(self.drop_rate), training=self.training)
        x = self.fc(x)
        return x



def resnet50_dla_l(**kwargs):
    print("Constructing resnet50_dla-light......")
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model

def resnet101_dla_l(**kwargs):
    print("Constructing resnet101_dla-light......")
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model

