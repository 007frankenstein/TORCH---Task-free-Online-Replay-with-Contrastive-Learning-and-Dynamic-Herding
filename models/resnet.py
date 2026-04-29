import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import relu, avg_pool2d

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out
    

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = relu(self.bn1(self.conv1(x)))
        out = relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = relu(out)
        return out

class cosLinear(nn.Module):
    def __init__(self, indim, outdim):
        super(cosLinear, self).__init__()
        self.L = nn.Linear(indim, outdim, bias = False)
        self.scale = 0.09



    def forward(self, x):
        x_norm = torch.norm(x, p=2, dim =1).unsqueeze(1).expand_as(x)
        x_normalized = x.div(x_norm+ 0.000001)

        L_norm = torch.norm(self.L.weight, p=2, dim =1).unsqueeze(1).expand_as(self.L.weight.data)
        weight_normalized = self.L.weight.div(L_norm + 0.000001)
        cos_dist = torch.mm(x_normalized,weight_normalized.transpose(0,1))
        scores = cos_dist / self.scale
        return scores
    
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes, nf, bias):
        super(ResNet, self).__init__()
        self.in_planes = nf
        self.conv1 = conv3x3(3, nf * 1)
        self.bn1 = nn.BatchNorm2d(nf * 1)
        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2) 
        self.linear = nn.Linear(nf * 8 * block.expansion, num_classes, bias=bias)

        self.in_features = nf * 8 * block.expansion
        self.out_features = num_classes

        self.pcrLinear = cosLinear(nf * 8 * block.expansion, num_classes)
        self.pcrLinear = cosLinear(160, num_classes)

        # self.projector = ProjectionHead(
        #     in_dim=self.in_features,
        #     hidden_dim=self.in_features,
        #     out_dim=128
        # )

        # self.classifier = nn.Linear(128, num_classes)


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    # def features(self, x):
    #     '''Features before FC layers'''
    #     out = F.relu(self.bn1(self.conv1(x))) # 64, 32, 32
    #     if hasattr(self, 'maxpool'):
    #         out = self.maxpool(out)
    #     out = self.layer1(out)  # -> 64, 32, 32
    #     out = self.layer2(out)  # -> 128, 16, 16
    #     out = self.layer3(out)  # -> 256, 8, 8
        
    #     # # Adaptive pooling
    #     # feature_before_pooling = self.layer4(out)  # -> 512, 4, 4
    #     # out = F.avg_pool2d(feature_before_pooling, feature_before_pooling.shape[2]) # -> 512, 1, 1
    #     # # out = F.adaptive_avg_pool2d(out, (1, 1))

    #     # # # Fixed pooling
    #     out = self.layer4(out)
    #     out = avg_pool2d(out, 4)
    #     # out = F.adaptive_avg_pool2d(out, (1, 1))
    #     out = out.view(out.size(0), -1) # -> 512
    #     return out

    def features(self, x):
        '''Features before FC layers'''
        out = relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = avg_pool2d(out, 4)
        # out = F.adaptive_avg_pool2d(out, (2, 2)) # Changed to (2,2) to get features of dim = (batch_size, 640)
        # out = F.adaptive_avg_pool2d(out, (1, 1)) # Changed to (2,2) to get features of dim = (batch_size, 640)
        # out = F.avg_pool2d(out, 2)
        # out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        return out
    
    # def cecr_forward(self, x):
    #     z = self.features(x)              # encoder output (θ)
    #     # z = F.normalize(z, dim=1)
    #     p = self.projector(z)             # projection (φ)
    #     # p = F.normalize(p, dim=1)
    #     # c = self.classifier(p)           # classifier output (W)
    #     return p #, c


    def logits(self, x):
        '''Apply the last FC linear mapping to get logits'''
        x = self.linear(x)
        return x

    def forward(self, x):
        out = self.features(x)
        # print(out.shape)
        logits = self.logits(out)
        return logits
    
    def pcrForward(self, x):
        out = self.features(x)
        # print(out.shape)
        logits = self.pcrLinear(out)
        return logits, out

def Reduced_ResNet18(nclasses, nf=20, bias=True):
    return ResNet(BasicBlock, [2, 2, 2, 2], nclasses, nf, bias)

class SupConResNet(nn.Module):
    def __init__(self, num_classes, dim_in=160, head='mlp', feat_dim=128):
        super(SupConResNet, self).__init__()
        self.encoder = Reduced_ResNet18(100)
        self.pcrLinear = self.encoder.pcrLinear
        
        if head == 'linear':
            self.head = nn.Linear(dim_in, feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )
        elif head == 'None':
            self.head = None
        else:
            raise NotImplementedError('head not supported: {}'.format(head))

    def forward(self, x):
        feat = self.encoder.features(x)
        if self.head:
            # feat = F.normalize(self.head(feat), dim=1)
            feat = F.normalize(self.head(feat.squeeze()), dim=1)
        else:
            feat = F.normalize(feat, dim=1)
        return feat

    def features(self, x):
        return self.encoder.features(x)
    
    def pcrForward(self, x):  # <-- Add this method
        return self.encoder.pcrForward(x)
    

##########################################################
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


class Encoder(nn.Module):
    def __init__(self, feat_dim=160, pretrained=False):
        super().__init__()

        backbone = resnet18(pretrained=pretrained)

        # Remove classifier
        self.backbone = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
            backbone.avgpool,
        )

        self.feat_dim = feat_dim

    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)   # [B, 512]
        return x


class ProjectionHead(nn.Module):
    def __init__(self, in_dim=160, hidden_dim=128, out_dim=128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)

class CECRModel(nn.Module):
    def __init__(self, feat_dim=160, proj_dim=128, hidden_dim=128):
        super().__init__()

        self.encoder = Encoder(feat_dim=feat_dim)
        self.projector = ProjectionHead(
            in_dim=feat_dim,
            hidden_dim=hidden_dim,
            out_dim=proj_dim
        )

    def encode(self, x):
        return self.encoder(x)

    def project(self, z):
        return self.projector(z)

    def forward(self, x):
        z = self.encoder(x)
        p = self.projector(z)
        return p

