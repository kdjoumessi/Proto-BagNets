import bagnets.pytorchnet
from torchvision import models

from .protoPNet_tools.bagnet_features import bagnet33_features, bagnet17_features, bagnet9_features
from .protoPNet_tools.resnet_features import resnet18_features, resnet34_features, resnet50_features, resnet101_features, resnet152_features
from .protoPNet_tools.densenet_features import densenet121_features, densenet161_features, densenet169_features, densenet201_features
from .protoPNet_tools.vgg_features import vgg11_features, vgg11_bn_features, vgg13_features, vgg13_bn_features, vgg16_features, vgg16_bn_features,\
                         vgg19_features, vgg19_bn_features

base_architecture_to_features = {'resnet18': resnet18_features,
                                 'resnet34': resnet34_features,
                                 'resnet50': resnet50_features,
                                 'resnet101': resnet101_features,
                                 'resnet152': resnet152_features,
                                 'densenet121': densenet121_features,
                                 'densenet161': densenet161_features,
                                 'densenet169': densenet169_features,
                                 'densenet201': densenet201_features,
                                 'vgg11': vgg11_features,
                                 'vgg11_bn': vgg11_bn_features,
                                 'vgg13': vgg13_features,
                                 'vgg13_bn': vgg13_bn_features,
                                 'vgg16': vgg16_features,
                                 'vgg16_bn': vgg16_bn_features,
                                 'vgg19': vgg19_features,
                                 'vgg19_bn': vgg19_bn_features,
                                 'bagnet33': bagnet33_features,
                                 'bagnet17': bagnet17_features,
                                 'bagnet9': bagnet9_features,
                                 }

BUILDER = {
    'vgg11': models.vgg11,
    'vgg13': models.vgg13,
    'vgg16': models.vgg16,
    'vgg19': models.vgg19,
    'resnet18': models.resnet18,
    'resnet34': models.resnet34,
    'resnet50': models.resnet50,
    'resnet101': models.resnet101,
    'resnet152': models.resnet152,
    'densenet121': models.densenet121,
    'densenet161': models.densenet161,
    'densenet169': models.densenet169,
    'densenet201': models.densenet201,
    'wide_resnet50': models.wide_resnet50_2,
    'wide_resnet101': models.wide_resnet101_2,
    'resnext50': models.resnext50_32x4d,
    'resnext101': models.resnext101_32x8d,
    'mobilenet': models.mobilenet_v2,
    'squeezenet': models.squeezenet1_1,
    'shufflenet_0_5': models.shufflenet_v2_x0_5,
    'shufflenet_1_0': models.shufflenet_v2_x1_0,
    'shufflenet_1_5': models.shufflenet_v2_x1_5,
    'shufflenet_2_0': models.shufflenet_v2_x2_0,
    'bagnet33': bagnets.pytorchnet.bagnet33,
}