import math
import bagnets
import torch.nn as nn
from torch.utils import model_zoo

model_dir = './pretrained_models'

__all__ = ['bagnet9', 'bagnet17', 'bagnet33']

model_urls = {
            'bagnet9': 'https://bitbucket.org/wielandbrendel/bag-of-feature-pretrained-models/raw/249e8fa82c0913623a807d9d35eeab9da7dcc2a8/bagnet8-34f4ccd2.pth.tar',
            'bagnet17': 'https://bitbucket.org/wielandbrendel/bag-of-feature-pretrained-models/raw/249e8fa82c0913623a807d9d35eeab9da7dcc2a8/bagnet16-105524de.pth.tar',
            'bagnet33': 'https://bitbucket.org/wielandbrendel/bag-of-feature-pretrained-models/raw/249e8fa82c0913623a807d9d35eeab9da7dcc2a8/bagnet32-2ddd53ed.pth.tar',
            }

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    # class attribute
    expansion = 1
    num_layers = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        # only conv with possibly not 1 stride
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        # if stride is not 1 then self.downsample cannot be None
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        # the residual connection
        out += identity
        out = self.relu(out)

        return out

    def block_conv_info(self):
        block_kernel_sizes = [3, 3]
        block_strides = [self.stride, 1]
        block_paddings = [1, 1]

        return block_kernel_sizes, block_strides, block_paddings


class Bottleneck(nn.Module):
    expansion = 4
    num_layers = 3 # @me

    def __init__(self, inplanes, planes, stride=1, downsample=None, kernel_size=1):
        super(Bottleneck, self).__init__()
        # print('Creating bottleneck with kernel size {} and stride {} with padding {}'.format(kernel_size, stride, (kernel_size - 1) // 2))
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False) # default stride=1
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=kernel_size, stride=stride, 
                               padding=0, bias=False) # changed padding from (kernel_size - 1) // 2
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)

        # if stride is not 1 then self.downsample cannot be None
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, **kwargs):
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
            residual = self.downsample(x)
        
        if residual.size(-1) != out.size(-1):
            diff = residual.size(-1) - out.size(-1)
            residual = residual[:,:,:-diff,:-diff]
        
        out += residual
        out = self.relu(out)

        return out

####
class BagNet(nn.Module):

    def __init__(self, block, layers, strides=[1, 2, 2, 2], kernel3=[0, 0, 0, 0], num_classes=1000, 
                 avg_pool=True, zero_init_residual=False):
        self.inplanes = 64
        super(BagNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=0.001)
        self.relu = nn.ReLU(inplace=True)

        self.kernel_sizes = [1, 3]
        self.strides = [1, 1]
        self.paddings = [0, 0]

        self.block = block     # @me
        self.layers = layers   # @me
        self.layer1 = self._make_layer(block, 64, layers[0], stride=strides[0], kernel3=kernel3[0], prefix='layer1')
        self.layer2 = self._make_layer(block, 128, layers[1], stride=strides[1], kernel3=kernel3[1], prefix='layer2')
        self.layer3 = self._make_layer(block, 256, layers[2], stride=strides[2], kernel3=kernel3[2], prefix='layer3')
        self.layer4 = self._make_layer(block, 512, layers[3], stride=strides[3], kernel3=kernel3[3], prefix='layer4')

        '''
        self.avgpool = nn.AvgPool2d(1, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.avg_pool = avg_pool
        self.block = block
        '''
        
        #@me: From BagNet
        # initialize the parameters
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        '''
        # @djoums: from ProtoPNet
        # initialize the parameters 
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        '''
        

        #@me: can be desactivated
        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, kernel3=0, prefix=''):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        kernel = 1 if kernel3 == 0 else 3
        layers.append(block(self.inplanes, planes, stride, downsample, kernel_size=kernel))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            kernel = 1 if kernel3 <= i else 3
            layers.append(block(self.inplanes, planes, kernel_size=kernel))

        #@me
        # keep track of every block's conv size, stride size, and padding size
        for each_block in layers:            
            for layer in each_block.children():
                if isinstance(layer, nn.Conv2d):
                    self.kernel_sizes.append(layer.kernel_size[0])
                    self.strides.append(layer.stride[0])
                    self.paddings.append(layer.padding[0])

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        '''
        if self.avg_pool:
            x = nn.AvgPool2d(x.size()[2], stride=1)(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
        else:
            x = x.permute(0,2,3,1)
            x = self.fc(x)
        '''
        return x
    
    def conv_info(self):
        return self.kernel_sizes, self.strides, self.paddings
    
    # @me
    def num_layers(self):
        '''
        the number of conv layers in the network, not counting the number
        of bypass layers
        '''

        return (self.block.num_layers * self.layers[0]
              + self.block.num_layers * self.layers[1]
              + self.block.num_layers * self.layers[2]
              + self.block.num_layers * self.layers[3]
              + 1)
    
    # @me
    def __repr__(self):
        template = 'bagnet_features' #'resnet{}_features'
        return template #.format(self.num_layers() + 1)
    

def bagnet33_features(pretrained=False, strides=[2, 2, 2, 1], **kwargs):
    """Constructs a Bagnet-33 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = BagNet(Bottleneck, [3, 4, 6, 3], strides=strides, kernel3=[1,1,1,1], **kwargs)
    if pretrained:
        my_dict = model_zoo.load_url(model_urls['bagnet33'], model_dir=model_dir)
        my_dict.pop('fc.weight')
        my_dict.pop('fc.bias')
        model.load_state_dict(my_dict, strict=False)
    return model

def bagnet17_features(pretrained=False, strides=[2, 2, 2, 1], **kwargs):
    """Constructs a Bagnet-17 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = BagNet(Bottleneck, [3, 4, 6, 3], strides=strides, kernel3=[1,1,1,0], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['bagnet17']))
    return model

def bagnet9_features(pretrained=False, strides=[2, 2, 2, 1], **kwargs):
    """Constructs a Bagnet-9 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = BagNet(Bottleneck, [3, 4, 6, 3], strides=strides, kernel3=[1,1,0,0], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['bagnet9']))
    return model

class Bagnet_model(nn.Module):
    def __init__(self, model):
        super(Bagnet_model, self).__init__()
        self.sequential = nn.Sequential(*model)
        
    def forward(self, x):
        out = self.sequential(x)
        return out
    
    def __repr__(self):        
        return 'bagnet_proto_backbone'

#########
def build_bagnet_model(cfg, pretrained=True):
    num_classes = cfg.data.num_classes
    tmp_model = bagnets.pytorchnet.bagnet33(pretrained=pretrained)
    tmp_model =  list(tmp_model.children())[:-2] 

    print('build the backbone bagnet from the pretraine model on ImageNet')
    model = Bagnet_model(tmp_model)
    
    return model

######
class Bagnet_v2(nn.Module):
    def __init__(self, model, num_classes=5):
        super(Bagnet_v2, self).__init__()
        self.sequential = nn.Sequential(*model)
        self.conv2 = nn.Conv2d(2048, num_classes, kernel_size=(1,1), stride=1)
        self.avgpool = nn.AvgPool2d(kernel_size=(1, 1), stride=(1,1), padding=0)

    def forward(self, x):
        x = self.sequential(x)
        x = self.conv2(x)        
        n, m = x.shape[2], x.shape[3]
        self.avgpool = nn.AvgPool2d(kernel_size=(n, m), stride=(1,1), padding=0)
        x = self.avgpool(x)
        out = x.view(x.shape[0], -1)
        return out  

if __name__ == '__main__':

    b9_features = bagnet9_features(pretrained=True)
    print(b9_features)

    b17_features = bagnet17_features(pretrained=True)
    print(b17_features)

    b33_features = bagnet33_features(pretrained=True)
    print(b33_features)

    bagnet_model = build_bagnet_model(pretrained=True)
    print(bagnet_model)
