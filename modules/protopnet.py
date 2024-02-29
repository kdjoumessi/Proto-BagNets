import torch
import torch.nn as nn
import torch.nn.functional as F

from .protoPNet_tools.bagnet_features import build_bagnet_model 
from .protoPNet_tools.resnet_features import build_resnet_model
from .protoPNet_tools.receptive_field import compute_proto_layer_rf_info_v2

class PPNet(nn.Module):

    def __init__(self, features, img_size, prototype_shape,
                 proto_layer_rf_info, num_classes, init_weights=True,
                 prototype_activation_function='log', cfg=None,
                 add_on_layers_type='bottleneck'):

        super(PPNet, self).__init__()
        self.proto_layer_rf_info = proto_layer_rf_info
        self.num_prototypes = prototype_shape[0]
        self.prototype_shape = prototype_shape        
        self.topk_k = cfg.prototype.topk
        self.num_classes = num_classes        
        self.img_size = img_size
        self.epsilon = 1e-4
        self.cfg = cfg        
        
        # prototype_activation_function could be 'log', 'linear',
        # or a generic function that converts distance to similarity score
        self.prototype_activation_function = prototype_activation_function

        '''
        Here we are initializing the class identities of the prototypes
        Without domain specific knowledge we allocate the same number of
        prototypes for each class
        '''
        assert(self.num_prototypes % self.num_classes == 0)
        # a onehot indication matrix for each prototype's class identity
        self.prototype_class_identity = torch.zeros(self.num_prototypes, self.num_classes)

        #self.sa_weight = torch.randn(self.num_prototypes)
        #self.sa_module = self.sa_weight.view(-1, self.num_classes)

        num_prototypes_per_class = self.num_prototypes // self.num_classes
        for j in range(self.num_prototypes):
            self.prototype_class_identity[j, j // num_prototypes_per_class] = 1

        # this has to be named features to allow the precise loading
        backbone = str(cfg.train.base_architecture).upper()
        if backbone.startswith('BAG'):            
            if cfg.train.build_backbone:
                print('build the backbone bagnet from scratch')
                self.features = features
            else:    
                print("build the backbone bagnet from the bagnet's library")            
                self.features = build_bagnet_model(cfg)
        else:
            if cfg.train.build_backbone:
                print('build the backbone resnet from scratch')
                self.features = features # features(torch.rand((1,3,512,512))).shape
            else:
                print('build the backbone resnet from the library')
                self.features = build_resnet_model(cfg)

        features_name = str(self.features).upper()
        # get the last output channel of the feature net => the channel number D 
        if features_name.startswith('VGG') or features_name.startswith('RES') or features_name.startswith('BAG'): 
            first_add_on_layer_in_channels = \
                [i for i in features.modules() if isinstance(i, nn.Conv2d)][-1].out_channels
        elif features_name.startswith('DENSE'):
            first_add_on_layer_in_channels = \
                [i for i in features.modules() if isinstance(i, nn.BatchNorm2d)][-1].num_features
        else:
            raise Exception('other base base_architecture NOT implemented')

        add_on_layers = []
        if add_on_layers_type == 'bottleneck':
            print('Add-on-layer type: bottleneck')
            current_in_channels = first_add_on_layer_in_channels
            while (current_in_channels > self.prototype_shape[1]) or (len(add_on_layers) == 0):
                current_out_channels = max(self.prototype_shape[1], (current_in_channels // 2))
                add_on_layers.append(nn.Conv2d(in_channels=current_in_channels,
                                               out_channels=current_out_channels,
                                               kernel_size=1))
                add_on_layers.append(nn.ReLU())
                add_on_layers.append(nn.Conv2d(in_channels=current_out_channels,
                                               out_channels=current_out_channels,
                                               kernel_size=1))
                if current_out_channels > self.prototype_shape[1]:
                    add_on_layers.append(nn.ReLU())
                else:
                    assert(current_out_channels == self.prototype_shape[1])
                    
                current_in_channels = current_in_channels // 2
            self.add_on_layers = nn.Sequential(*add_on_layers)
        else:
            add_on_layers = [nn.Conv2d(in_channels=first_add_on_layer_in_channels, out_channels=self.prototype_shape[1], kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=self.prototype_shape[1], out_channels=self.prototype_shape[1], kernel_size=1)]
            
        add_on_layers.append(nn.Sigmoid())
        self.add_on_layers = nn.Sequential(*add_on_layers)

        self.prototype_vectors = nn.Parameter(torch.rand(self.prototype_shape), requires_grad=True)
        self.ones = nn.Parameter(torch.ones(self.prototype_shape), requires_grad=False)

        # f = feature + ReLU(1x1 conv) Sigmoid(1x1 conv) + proto_layer + FCL
        # final FCL / SA
        if cfg.train.fc_classification_layer:
            print('FCL layer module for classification')
            self.last_layer = nn.Linear(self.num_prototypes, self.num_classes, bias=False) 
        else: # SA
            print('Score aggregation (SA) module for classification')
            if cfg.train.random_SA_init:
                print('SA initialisation: torch.randn')
                self.last_layer = nn.Parameter(torch.randn(num_prototypes_per_class, self.num_classes), requires_grad=True)
            else:
                print('SA initialisation: torch.ones')
                self.last_layer = nn.Parameter(torch.ones(num_prototypes_per_class, self.num_classes), requires_grad=True)
        
        if init_weights:
            self._initialize_weights(cfg)

    def conv_features(self, x): # apply the feature extractor + add-on-layer => f(x)
        '''
        the feature input to prototype layer
        '''
        x = self.features(x) # z = f(x)
        x = self.add_on_layers(x)
        return x
    
    def set_topk_k(self, topk_k):
        '''set the topk_k'''
        self.topk_k = topk_k

    @staticmethod
    def _weighted_l2_convolution(input, filter, weights):
        '''
        input of shape N * c * h * w
        filter of shape P * c * h1 * w1
        weight of shape P * c * h1 * w1
        '''
        input2 = input ** 2
        input_patch_weighted_norm2 = F.conv2d(input=input2, weight=weights)

        filter2 = filter ** 2
        weighted_filter2 = filter2 * weights
        filter_weighted_norm2 = torch.sum(weighted_filter2, dim=(1, 2, 3))
        filter_weighted_norm2_reshape = filter_weighted_norm2.view(-1, 1, 1)

        weighted_filter = filter * weights
        weighted_inner_product = F.conv2d(input=input, weight=weighted_filter)

        # use broadcast
        intermediate_result = \
            - 2 * weighted_inner_product + filter_weighted_norm2_reshape
        # x2_patch_sum and intermediate_result are of the same shape
        distances = F.relu(input_patch_weighted_norm2 + intermediate_result)

        return distances

    def _l2_convolution(self, x): # take the feature map and output
        '''
        apply self.prototype_vectors as l2-convolution filters on input x
        '''
        x2 = x ** 2
        x2_patch_sum = F.conv2d(input=x2, weight=self.ones)

        p2 = self.prototype_vectors ** 2
        p2 = torch.sum(p2, dim=(1, 2, 3))
        # p2 is a vector of shape (num_prototypes,)
        # then we reshape it to (num_prototypes, 1, 1)
        p2_reshape = p2.view(-1, 1, 1)

        xp = F.conv2d(input=x, weight=self.prototype_vectors)
        intermediate_result = - 2 * xp + p2_reshape  # use broadcast
        # x2_patch_sum and intermediate_result are of the same shape
        distances = F.relu(x2_patch_sum + intermediate_result)
        return distances

    def prototype_distances(self, x):
        '''
        x is the raw input
        '''
        conv_features = self.conv_features(x)
        distances = self._l2_convolution(conv_features)
       
        return distances

    def distance_2_similarity(self, distances):
        if self.prototype_activation_function == 'log':
            return torch.log((distances + 1) / (distances + self.epsilon))
        elif self.prototype_activation_function == 'linear':
            return -distances
        else:
            return self.prototype_activation_function(distances)

    def forward(self, x):
        distances = self.prototype_distances(x)
        l1_norm = torch.norm(distances) #.detach()
        '''
        we cannot refactor the lines below for similarity scores
        because we need to return min_distances
        '''
        if self.topk_k:
            _distances = distances.view(distances.shape[0], distances.shape[1], -1)
            top_k_neg_distances, _ = torch.topk(-_distances, self.topk_k)
            closest_k_distances = - top_k_neg_distances
            min_distances = F.avg_pool1d(closest_k_distances, 
                                         kernel_size=closest_k_distances.shape[2]).view(-1, self.num_prototypes)
            prototype_activations = self.distance_2_similarity(min_distances)
        else:
            # global min pooling
            min_distances = -F.max_pool2d(-distances,kernel_size=(distances.size()[2], distances.size()[3]))
            min_distances = min_distances.view(-1, self.num_prototypes)       
            prototype_activations = self.distance_2_similarity(min_distances)

        if self.cfg.train.fc_classification_layer:
            logits = self.last_layer(prototype_activations)
        else:
            n = self.cfg.data.num_classes
            bs = prototype_activations.shape[0]
            prototype_activations = prototype_activations.view(bs, -1, n)
            logits = (self.last_layer * prototype_activations).sum(dim=1)

        return logits, min_distances, l1_norm 

    def push_forward(self, x):
        '''this method is needed for the pushing operation'''
        conv_output = self.conv_features(x) # feature + add-on
        distances = self._l2_convolution(conv_output)
        return conv_output, distances

    def prune_prototypes(self, prototypes_to_prune):
        '''
        prototypes_to_prune: a list of indices each in
        [0, current number of prototypes - 1] that indicates the prototypes to
        be removed
        '''
        prototypes_to_keep = list(set(range(self.num_prototypes)) - set(prototypes_to_prune))

        self.prototype_vectors = nn.Parameter(self.prototype_vectors.data[prototypes_to_keep, ...],
                                              requires_grad=True)

        self.prototype_shape = list(self.prototype_vectors.size())
        self.num_prototypes = self.prototype_shape[0]

        # changing self.last_layer in place
        # changing in_features and out_features make sure the numbers are consistent
        self.last_layer.in_features = self.num_prototypes
        self.last_layer.out_features = self.num_classes
        self.last_layer.weight.data = self.last_layer.weight.data[:, prototypes_to_keep]

        # self.ones is nn.Parameter
        self.ones = nn.Parameter(self.ones.data[prototypes_to_keep, ...],
                                 requires_grad=False)
        # self.prototype_class_identity is torch tensor
        # so it does not need .data access for value update
        self.prototype_class_identity = self.prototype_class_identity[prototypes_to_keep, :]

    def __repr__(self):
        rep = (
            'PPNet(\n'
            '\tfeatures: {},\n'
            '\timg_size: {},\n'
            '\tprototype_shape: {},\n'
            '\tproto_layer_rf_info: {},\n'
            '\tnum_classes: {},\n'
            '\tepsilon: {}\n'
            ')'
        )

        return rep.format(self.features,
                          self.img_size,
                          self.prototype_shape,
                          self.proto_layer_rf_info,
                          self.num_classes,
                          self.epsilon)

    def set_last_layer_incorrect_connection(self, incorrect_strength):
        '''
        the incorrect strength will be actual strength if -0.5 then input -0.5
        '''
        positive_one_weights_locations = torch.t(self.prototype_class_identity)
        negative_one_weights_locations = 1 - positive_one_weights_locations

        correct_class_connection = 1
        incorrect_class_connection = incorrect_strength
        self.last_layer.weight.data.copy_(
            correct_class_connection * positive_one_weights_locations
            + incorrect_class_connection * negative_one_weights_locations)

    def _initialize_weights(self, cfg):
        for m in self.add_on_layers.modules():
            if isinstance(m, nn.Conv2d):
                # every init technique has an underscore _ in the name
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if cfg.train.fc_classification_layer:
            self.set_last_layer_incorrect_connection(incorrect_strength=-0.5)
        else:
            if cfg.train.fan_out_sa_init:
                print('SA initialisation: fan_out with the other layer')
                nn.init.kaiming_normal_(self.last_layer, mode='fan_out')


def construct_PPNet(cfg, base_architecture, pretrained=True, img_size=224,
                    prototype_shape=(2000, 512, 1, 1), num_classes=200,
                    prototype_activation_function='log',
                    add_on_layers_type='bottleneck',
                    backbone=None):
    features = backbone[base_architecture](pretrained=pretrained)
    layer_filter_sizes, layer_strides, layer_paddings = features.conv_info()
    
    proto_layer_rf_info = compute_proto_layer_rf_info_v2(img_size=img_size,
                                                         layer_filter_sizes=layer_filter_sizes,
                                                         layer_strides=layer_strides,
                                                         layer_paddings=layer_paddings,
                                                         prototype_kernel_size=prototype_shape[2])
    return PPNet(features=features,
                 img_size=img_size,
                 prototype_shape=prototype_shape,
                 proto_layer_rf_info=proto_layer_rf_info,
                 num_classes=num_classes,
                 init_weights=True, cfg=cfg,
                 prototype_activation_function=prototype_activation_function,
                 add_on_layers_type=add_on_layers_type)