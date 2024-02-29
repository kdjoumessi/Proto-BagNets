import os
import cv2
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import feature, transform

from PIL import Image
from torchvision import transforms

def load_dataset():
    '''
        Load and return the test and validation dataset into a pandas dataframe
    '''
    root = "/gpfs01/berens/data/data/DR/Kaggle_DR_resized/"
    train_file_name  = 'kaggle_gradable_train_new_qual_eval.csv'
    test_file_name = 'kaggle_gradable_test_new_qual_eval.csv'
    val_file_name  = 'kaggle_gradable_val_new_qual_eval.csv'

    df_train  = pd.read_csv(os.path.join(root, train_file_name))
    df_val  = pd.read_csv(os.path.join(root, val_file_name))
    df_test = pd.read_csv(os.path.join(root, test_file_name))
    
    return df_train, df_val, df_test


def load_EyePAC(cfg, path, OCT=False, filename=None, df=None):
    '''
    return the corresponding image (normalized and without) from the dataframe: => (batch_size, C, H, W).

        Parameters:
            - path (str): image path location
            - df: dataframe
        
        Returns:
            - PIL normalize (in [0, 1]) with shape (b, C,H,W)  
            - PIL unormalize image
    '''
    normalization = [
        transforms.Resize((cfg.data.input_size)), # (cfg.data.input_size, cfg.data.input_size)
        transforms.ToTensor(),
        transforms.CenterCrop(cfg.data.input_size),
        transforms.Normalize(cfg.data.mean, cfg.data.std)]
    
    test_preprocess_norm = transforms.Compose(normalization)
    test_preprocess = transforms.Compose(normalization[:-1])
    
    if filename:
        file_paths = os.path.join(path, filename) 
        pil_img = Image.open(file_paths)

        img_norm = test_preprocess_norm(pil_img)
        img = test_preprocess(pil_img)
        
        img = torch.unsqueeze(img, dim=0)
        img_norm = torch.unsqueeze(img_norm, dim=0)
    else:
        file_paths = [os.path.join(path, file) for file in df.filename]          
        if OCT:
            file_paths = [os.path.join(path, file) for file in df.filenames] 
            
        pil_img = [Image.open(file) for file in file_paths]

        img_norm = [test_preprocess_norm(pil_img[i]) for i in range(len(pil_img))]
        img = [test_preprocess(pil_img[i]) for i in range(len(pil_img))]
        img_norm = torch.stack(img_norm)
        img = torch.stack(img)           
    return img_norm, img


def get_HD_similarity_map(image, proto_SimMap, proto_max_pool_simScores, l2_dist_sim, j=0, k=0, yhat=0, dilation=0.5, percentile=99, cmap='RdBu_r',
                         cv2_upsample=False, softmax=True):
    ''' proto_max_pool_simScores
        description: return the similarity map of the ProtoPNet in the BagNet style (heatmap overlay on the input edges)

        parameters:
            - image (torch.tensor [bs, H,W,C]): original images
            - proto_SimScore (np.array => [bs, #proto, NxM]):  where #proto is the number of prototype and NxM the spatial dim of the simMap
            - j (int): the index of the image of interest withing the batch
            - k (int): proto choose for max and min color map value
            - yhat (int): predicted class
            - proto_max_pool_simScores (np.array [bs, #proto]): the final proto acyivation after MaxPool of NxM similarity map
        output:
            - 
    '''
    # detach the corresponding values from the batch
    np_img = image.numpy()
    size = np_img.shape[1]
    protoSimMap = proto_SimMap[j]
    max_pool_sim_proto_scores = proto_max_pool_simScores[j]
    max_l2_dist_sim = l2_dist_sim[j]
    
    # upsampling the heatmap from the low resolution to the input size (512x512)        
    np_img = np_img.transpose([0,3,1,2]) 
    m = torch.nn.Upsample(size=[size, size], mode='bilinear', align_corners=True)
    
    dx, dy = 0.05, 0.05
    def get_proto_wise_simMap(): # for each proto => [n_proto, 512, 512]
        # from low to high resolution feature maps
        tmp_fmap_list = []
        for i in range(len(protoSimMap)):
            tmp_fmaps = np.exp(protoSimMap[i]) / np.exp(protoSimMap[i]).sum() if softmax else protoSimMap[i]
            if cv2_upsample:
                #print(tmp_fmaps.shape)
                upsampled_act_img = cv2.resize(tmp_fmaps, dsize=(size, size), interpolation=cv2.INTER_CUBIC) # cv2.INTER_AREA, INTER_CUBIC
                tmp_hd_simMap = torch.from_numpy(upsampled_act_img)
                #print(upsampled_act_img.shape)
            else:
                tmp_simScores = tmp_fmaps[np.newaxis, np.newaxis, :, :] # adding two new dimension to the low resolution heatmap
                tmp_upsample = m(torch.from_numpy(tmp_simScores))
                tmp_hd_simMap = tmp_upsample[0,0]
            tmp_fmap_list.append(tmp_hd_simMap)
        fmaps = torch.stack(tmp_fmap_list, dim=0)
        return fmaps
    
    def bagnet_heatmap_template(scores, image, k=k):   # image[0]  
        proto_heatmap = {}
        score = scores[k]
        #print(score.shape)
        xx = np.arange(0.0, score.shape[1], dx)
        yy = np.arange(0.0, score.shape[0], dy)
        xmin, xmax, ymin, ymax = np.amin(xx), np.amax(xx), np.amin(yy), np.amax(yy)
        extent = xmin, xmax, ymin, ymax
        cmap_original = plt.get_cmap('Greys_r').copy()
        cmap_original.set_bad(alpha=0)
        overlay = None
        
        original_greyscale = np.mean(image.transpose([1,2,0]), axis=-1)
        in_image_upscaled = transform.rescale(original_greyscale, dilation, mode='constant', channel_axis=False, anti_aliasing=True)
        edges = feature.canny(in_image_upscaled).astype(float)
        edges[edges < 0.5] = np.nan
        edges[:5, :] = np.nan
        edges[-5:, :] = np.nan
        edges[:, :5] = np.nan
        edges[:, -5:] = np.nan
        overlay = edges
        
        abs_max = np.percentile(np.abs(score), percentile)
        abs_min = abs_max
        for i in range(len(scores)):
            proto_heatmap[i] = {'heatmap': scores[i], 'overlay': overlay, 'extent': extent, 'cmap': cmap, 
                                'cmap_original': cmap_original, 'vmin': abs_min, 'vmax': abs_max, 'max_pool_proto_score': (max_l2_dist_sim[i], max_pool_sim_proto_scores[i])}
        return proto_heatmap
    
    proto_fmaps = get_proto_wise_simMap()
    proto_heat = bagnet_heatmap_template(proto_fmaps, np_img[j])
    
    return proto_heat 


def get_HD_similarity_map_v2(image, proto_SimMap, yhat=1, dilation=0.5, percentile=99, cmap='RdBu_r', cv2_upsample=False, softmax=True):
    ''' proto_max_pool_simScores
        description: return the similarity map of the ProtoPNet in the BagNet style (heatmap overlay on the input edges)

        parameters:
            - image (torch.tensor [bs, H,W,C]): original images
            - proto_SimScore (np.array => [bs, #proto, NxM]):  where #proto is the number of prototype and NxM the spatial dim of the simMap
            - j (int): the index of the image of interest withing the batch
            - k (int): proto choose for max and min color map value
            - yhat (int): predicted class
            - proto_max_pool_simScores (np.array [bs, #proto]): the final proto acyivation after MaxPool of NxM similarity map
        output:
            - 
    '''
    # detach the corresponding values from the batch
    np_img = image.numpy()
    size = np_img.shape[1]
    
    # upsampling the heatmap from the low resolution to the input size (512x512)        
    np_img = np_img.transpose([0,3,1,2]) 
    m = torch.nn.Upsample(size=[size, size], mode='bilinear', align_corners=True)
    
    dx, dy = 0.05, 0.05
    def get_proto_wise_simMap(): # for each proto => [n_proto, 512, 512]
        # from low to high resolution feature maps
        tmp_fmap_list = []
        for i in range(len(proto_SimMap[0])):
            tmp_fmaps =  proto_SimMap[0][i]
            tmp_simScores = tmp_fmaps[np.newaxis, np.newaxis, :, :] # adding two new dimension to the low resolution heatmap
            tmp_upsample = m(torch.from_numpy(tmp_simScores))
            tmp_hd_simMap = tmp_upsample[0,0]
            tmp_fmap_list.append(tmp_hd_simMap)
        fmaps = torch.stack(tmp_fmap_list, dim=0)
        return fmaps
    
    def bagnet_heatmap_template(scores, image): 
        proto_heatmap = {}
        score = scores[0]
        xx = np.arange(0.0, score.shape[1], dx)
        yy = np.arange(0.0, score.shape[0], dy)
        xmin, xmax, ymin, ymax = np.amin(xx), np.amax(xx), np.amin(yy), np.amax(yy)
        extent = xmin, xmax, ymin, ymax
        cmap_original = plt.get_cmap('Greys_r').copy()
        cmap_original.set_bad(alpha=0)
        overlay = None
        
        original_greyscale = np.mean(image.transpose([1,2,0]), axis=-1)
        in_image_upscaled = transform.rescale(original_greyscale, dilation, mode='constant', channel_axis=False, anti_aliasing=True)
        edges = feature.canny(in_image_upscaled).astype(float)
        edges[edges < 0.5] = np.nan
        edges[:5, :] = np.nan
        edges[-5:, :] = np.nan
        edges[:, :5] = np.nan
        edges[:, -5:] = np.nan
        overlay = edges
        
        abs_max = np.percentile(np.abs(score), percentile)
        abs_min = abs_max
        for i in range(len(scores)):
            proto_heatmap[i] = {'heatmap': scores[i], 'overlay': overlay, 'extent': extent, 'cmap': cmap, 'cmap_original': cmap_original, 'vmin': abs_min, 'vmax': abs_max}
        return proto_heatmap
    
    proto_fmaps = get_proto_wise_simMap()
    proto_heat = bagnet_heatmap_template(proto_fmaps, np_img[0])
    
    return proto_heat