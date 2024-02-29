import os
import math
import torch
import matplotlib
import matplotlib.pyplot as plt 
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable

from utils.utils import get_HD_similarity_map_v2

# (sim_map, min_distances, proto_max_pool_similarity, pred_class, k, plot_proto=False, path=None, size=(20,20), fs=12):
def plot_topk_proto(dic_proto, min_dist, np_img, paths, size=(20,20), fs=12):
    plt.rcParams['font.size'] = f'{fs}'  
    
    n = len(dic_proto)
    fig = plt.subplots(layout='constrained', figsize=size)
    
    list_proto = []
    label = []
    k = int(len(min_dist) / len(dic_proto))
    j = 0
    for i in range(n):
        list_proto.append(plt.imread(f'{paths}{i}.png'))
        label.append(f'proto_{i}')
        
        proto_i = dic_proto[i]
        for p in proto_i: # [  0, 144, 241, 208, 305,   1]
            list_proto.append(np_img[p[1]:p[2], p[3]:p[4], :])
            label.append(round(min_dist[j], 7))
            j += 1
    
    for i in range(len(list_proto)):
        plt.subplot(n, k+1, i+1)
        plt.title(label[i])
        plt.imshow(list_proto[i])
        plt.axis('off')

def plot_proto_log_similary_map_(sim_map, min_distances, proto_max_pool_similarity, pred_class, k, plot_proto=False, path=None, size=(20,20), fs=12):
    # log similarity map, l2_min_distance, log_similarity
    plt.rcParams['font.size'] = f'{fs}'  
    # num proto / num class
    n = pred_class*k # init proto
    fig = plt.subplots(layout='constrained', figsize=size)
    
    proto, s_map = [], []
    for i in range(n, n+k):
        proto.append(plt.imread(f'{path}{i}.png'))
        s_map.append(sim_map[0,i])
    
    j = 0
    if plot_proto:
        proto_smap = proto + s_map
        for i in range(len(proto_smap)):
            plt.subplot(2, k, i+1)
            if i < k:
                plt.title(f'Proto {n+i}, Score: {round(min_distances[0, i+k].cpu().item(), 4), round(proto_max_pool_similarity[0,i+k].item(), 3)}')    
            plt.imshow(proto_smap[i])
            if i !=0:
                plt.axis('off')
    else:
        for i in range(n, n+k):
            plt.subplot(1, k+1, j+1) # (row nber, col nber, image nber)
            plt.imshow(sim_map[0,i])        
            plt.title(f'Proto {i}, Score: {round(min_distances[0, i+k].cpu().item(), 4), round(proto_max_pool_similarity[0,i+k].item(), 3)}') #SimMap-Score: (l2-dist, log-dist)
            if j !=0:
                plt.axis('off')
            j = j+1

def plot_proto_bag_heatmap(proto_heat, n_proto, pred_class, init, size = (21,21), fs=12):

    plt.rcParams['font.size'] = f'{fs}'
    fig = plt.figure(figsize=size, layout='constrained')
    j=0
    init = n_proto * pred_class # init_simMap_nber
    ncol = n_proto + 1
    vmax = proto_heat[0]['vmax']
    for i in range(n_proto):
        j += 1
        ax = fig.add_subplot(n_proto, ncol, j)
        heatmap = proto_heat[init+i]
        img = ax.imshow(heatmap['heatmap'], extent=heatmap['extent'], interpolation='none', cmap=heatmap['cmap'], vmin=-vmax, vmax=vmax)
        ax.imshow(heatmap['overlay'], extent=heatmap['extent'], interpolation='none', cmap=heatmap['cmap_original'], alpha=.25)

        plt.title(f"proto {init+i}, Score: {(heatmap['max_pool_proto_score'][1], heatmap['max_pool_proto_score'][0])}") #MaxPool-SimScore
        ax.axis('off')

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(img, cax=cax)

def plot_proto_bag_heatmap_v2(proto_heat, n_proto, pred_class, init, imgs=(True, None), size = (21,21), fs=12):

    plt.rcParams['font.size'] = f'{fs}'
    fig = plt.figure(figsize=size, layout='constrained')
    j=0
    init = n_proto * pred_class # init_simMap_nber
    ncol = (n_proto + 1) if imgs[0] else (n_proto + 2)
    vmax = proto_heat[0]['vmax']
    l=0
    
    if imgs[0]:
        im = imgs[1].numpy()
        ax = fig.add_subplot(n_proto, ncol, 1)
        ax.imshow(im[0])
        ax.axis('off')
        l = 1
        
    for i in range(n_proto):
        j += 1
        ax = fig.add_subplot(n_proto, ncol, j+l)
        heatmap = proto_heat[init+i]
        img = ax.imshow(heatmap['heatmap'], extent=heatmap['extent'], interpolation='none', cmap=heatmap['cmap'], vmin=-vmax, vmax=vmax)
        ax.imshow(heatmap['overlay'], extent=heatmap['extent'], interpolation='none', cmap=heatmap['cmap_original'], alpha=.25)

        if imgs[0]:
            heat = torch.tensor(heatmap['max_pool_proto_score'][0])
            plt.title(f"{round(heatmap['max_pool_proto_score'][0].item(), 3)}")
        else:
            plt.title(f"proto {init+i}, Score: {(heatmap['max_pool_proto_score'][1], heatmap['max_pool_proto_score'][0])}") #MaxPool-SimScore
        ax.axis('off')

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(img, cax=cax)
    
def plot_proto_bag_heatmap_v3(proto_heat, n_proto, pred_class, init, l2_dist, log_dist, nrow=1, imgs=(True, None), size = (21,21), 
                              fs=12, alpha=0.25, vmax_=None):
    plt.rcParams['font.size'] = f'{fs}'
    fig = plt.figure(figsize=size, layout='constrained')
    j, l= 0,0
    init = n_proto * pred_class # init_simMap_nber
    ncol = (n_proto + 1) if imgs[0] else (n_proto + 2)
    
    if vmax_: 
        vmax = vmax_
    else:
        vmax = proto_heat[0]['vmax']
    
    if imgs[0]:
        im = imgs[1].numpy()
        ax = fig.add_subplot(nrow, ncol, 1)
        ax.imshow(im[0])
        ax.axis('off')
        l = 1
    
    N = pred_class*n_proto
    for i in range(init, init+n_proto):
        j += 1
        ax = fig.add_subplot(nrow, ncol, j+l)
        heatmap = proto_heat[i]
        img = ax.imshow(heatmap['heatmap'], extent=heatmap['extent'], interpolation='none', cmap=heatmap['cmap'], vmin=-vmax, vmax=vmax)
        ax.imshow(heatmap['overlay'], extent=heatmap['extent'], interpolation='none', cmap=heatmap['cmap_original'], alpha=alpha)

        heat = torch.tensor(heatmap['max_pool_proto_score'][0])
        plt.title(f"{round(l2_dist[0][i].item(), 4), round(log_dist[0][i].item(), 3)}")
        ax.axis('off')

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(img, cax=cax)

def plot_simple_heatmap(proto_heat, pred_class, n_proto, init, size = (21,21), fs=12):
    plt.rcParams['font.size'] = f'{fs}'

    fig = plt.figure(figsize=size, layout='constrained')
    j=0
    init = n_proto * pred_class # init_simMap_nber
    ncol = n_proto + 1
    for i in range(n_proto):
        j += 1
        ax = fig.add_subplot(n_proto, ncol, j)
        heatmap = proto_heat[init+i]
        ax.imshow(heatmap['heatmap'])
        ax.axis('off')

    divider = make_axes_locatable(ax)

def plot_high_activate_patch(np_plot_img, proto_bb, pred_class, n_proto, size=(20,20), fs=12):
    k = pred_class*n_proto
    plt.rcParams['font.size'] = f'{fs}'
    j=0
    fig = plt.subplots(layout='constrained', figsize=size)
    for i in range(k, k+n_proto): #len(proto_bb)
        proto = proto_bb[i]
        plt.subplot(1, k+1, j+1) # (row nber, col nber, image nber)
        tmp = np_plot_img[0][proto[1]:proto[2], proto[3]:proto[4]:]
        plt.imshow(tmp)
        plt.title(f'proto {i}, {tmp.shape[:2]}')
        plt.axis('off')
        j += 1

    
def plot_bounding_box(np_plot_img, proto_bb, top_proto_bb, pred_class, n_proto, top_proto=False, 
                        size=(40,40), fs=12, line=1):
    j=0
    k = pred_class*n_proto
    plt.rcParams['font.size'] = f'{fs}'
    fig = plt.figure(layout='constrained', figsize=size)
    for i in range(k, k+n_proto):
        coord = proto_bb[i]
        ax = fig.add_subplot(1, k+1, j+1)
        #plt.subplot(1, k+1, j+1) # (row nber, col nber, image nber)
        bottom_left = (coord[3], coord[1])
        width = coord[2]-coord[1]
        height = coord[4]-coord[3]
        #print(bottom_left, width, height)

        rect = matplotlib.patches.Rectangle(bottom_left, height, width, linewidth=line, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

        if top_proto:
            coord = top_proto_bb[i]
            rect = matplotlib.patches.Rectangle((coord[3], coord[1]), coord[4]-coord[3], coord[2]-coord[1], 
            linewidth=line, edgecolor='g', facecolor='none')
            ax.add_patch(rect)

        ax.imshow(np_plot_img[0])
        plt.axis('off')
        j += 1
        
def plot_bounding_box_all(np_plot_img, proto_bb, top_proto_bb, pred_class, top_proto=False, nrow=4, ncol=4,
                        size=(40,40), fs=12, line=1):
    j=0
    k = len(proto_bb)
    plt.rcParams['font.size'] = f'{fs}'
    fig = plt.figure(layout='constrained', figsize=size)
    for i in range(k):
        coord = proto_bb[i]
        ax = fig.add_subplot(nrow, ncol, i+1)
        bottom_left = (coord[3], coord[1])
        width = coord[2]-coord[1]
        height = coord[4]-coord[3]
        #print(bottom_left, width, height)

        rect = matplotlib.patches.Rectangle(bottom_left, height, width, linewidth=line, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

        if top_proto:
            coord = top_proto_bb[i]
            rect = matplotlib.patches.Rectangle((coord[3], coord[1]), coord[4]-coord[3], coord[2]-coord[1], 
            linewidth=line, edgecolor='g', facecolor='none')
            ax.add_patch(rect)

        ax.imshow(np_plot_img[0])
        plt.axis('off')
        
def plot_bin_topk_bounding_box(np_plot_img, proto_bb_dic, ncol=5, size=(40,40), fs=12, line=1, nrow=1):
    j=0
    plt.rcParams['font.size'] = f'{fs}'
    fig = plt.figure(layout='constrained', figsize=size)
    k=1
    for i in range(5, 10):
        coord_i = proto_bb_dic[i]
        #ax = fig.add_subplot(nrow, ncol, i+1)
        ax = fig.add_subplot(nrow, ncol, k)
        #plt.subplot(nrow, ncol, i+1)
        #ax = plt.gca()
        
        for j in range(len(coord_i)): 
            coord = coord_i[j]
            bottom_left = (coord[3], coord[1])
            width = coord[2]-coord[1]
            height = coord[4]-coord[3]

            rect = matplotlib.patches.Rectangle(bottom_left, height, width, linewidth=line, edgecolor='r', facecolor='none')
            ax.add_patch(rect)

        ax.imshow(np_plot_img[0])
        k += 1
        plt.axis('off')
        
def plot_bin_topk_bounding_box_v2(np_plot_img, proto_bb_dic, ncol=5, size=(40,40), fs=12, line=1, nrow=1):
    j=0
    plt.rcParams['font.size'] = f'{fs}'
    fig = plt.figure(layout='constrained', figsize=size)
    k=1
    for i in range(5, 10):
        coord_i = proto_bb_dic[i]
        #ax = fig.add_subplot(nrow, ncol, i+1)
        ax = fig.add_subplot(nrow, ncol, k)
        #plt.subplot(nrow, ncol, i+1)
        #ax = plt.gca()
        
        for j in range(len(coord_i)): 
            coord = coord_i[j]
            bottom_left = (coord[3], coord[1])
            width = coord[2]-coord[1]
            height = coord[4]-coord[3]

            rect = matplotlib.patches.Rectangle(bottom_left, height, width, linewidth=line, edgecolor='r', facecolor='none')
            ax.add_patch(rect)

        ax.imshow(np_plot_img[0])
        k += 1
        plt.axis('off')

def plot_learned_proto_full_img(proto_from_training, n_proto, init, size=(20,20), fs=12, shape=True):
    plt.rcParams['font.size'] = f'{fs}'
    fig = plt.figure(figsize=(size), layout='constrained')
    j=0
     # init_simMap_nber
    ncol = n_proto + 1
    for i in range(n_proto):
        j += 1
        k = i + init
        tmp = proto_from_training[k]
        ax = fig.add_subplot(n_proto, ncol, j)
        ax.imshow(tmp)
        if shape:
            plt.title(f'img: {k}, {tmp.shape[:2]}')
        else:
            plt.title(f'img: {k}')
        ax.axis('off')

def plot_proto(path, simMap=(None, None, None), title=None, cols=5, n=20, fs=14, bb=False, bb2=False, 
                bb_rf=None, top_bb=None, line=2, color=['r', 'b'], shape=False, axis=True, size=(10,10)):
    
    plt.rcParams['font.size'] = f'{fs}'
    fig = plt.subplots(layout='constrained', figsize=size)
    n_rows = math.ceil(n/cols)
    
    if title:
        plt.title(title)
    
    for i in range(n):
        plt.subplot(n_rows, cols, i+1)
        ax = plt.gca()
        img = plt.imread(f'{path}{i}.png')
        title = f'proto-{i}'
        ax.imshow(img)
        
        if bb:
            rect = patches.Rectangle((bb_rf[i][3], bb_rf[i][1]), bb_rf[i][2]-bb_rf[i][1], bb_rf[i][4]-bb_rf[i][3], linewidth=2, edgecolor=color[0], facecolor='none')
            #ax = plt.gca()
            ax.add_patch(rect)
        if bb2:
            rect = patches.Rectangle((top_bb[i][3], top_bb[i][1]), top_bb[i][2]-top_bb[i][1], top_bb[i][4]-top_bb[i][3], linewidth=2.5, edgecolor=color[1], facecolor='none')
            #ax = plt.gca()
            ax.add_patch(rect)
            
        if shape:
            title = f'{i}: {img.shape[:2]}'
        if simMap[0]:                    
            #title = f'({i}, {round(simMap[1][0][i].cpu().item(), 3)}, {round(simMap[2][i].item(), 3)})' #
            title = f'({i}, {round(simMap[2][i].item(), 3)})'  
            pass
            
        ax.set_title(title)
        if not axis:
            ax.axis('off')

def plot_proto_v2(path, simMap=(None, None, None), title=None, cols=5, n=20, fs=14, bb=False, bb2=False, proto_title=False, dic_title=None,
                bb_rf=None, top_bb=None, line=2, color=['r', 'b'], shape=False, axis=True, size=(10,10)):
    
    plt.rcParams['font.size'] = f'{fs}'
    fig = plt.subplots(layout='constrained', figsize=size)
    n_rows = math.ceil(n/cols)
    
    if title:
        plt.title(title)
    
    for i in range(n):
        plt.subplot(n_rows, cols, i+1)
        ax = plt.gca()
        img = plt.imread(f'{path}{i}.png')
        title = f'proto-{i}'
        ax.imshow(img)
        
        if bb:
            rect = patches.Rectangle((bb_rf[i][3], bb_rf[i][1]), bb_rf[i][2]-bb_rf[i][1], bb_rf[i][4]-bb_rf[i][3], linewidth=2, edgecolor=color[0], facecolor='none')
            #ax = plt.gca()
            ax.add_patch(rect)
        if bb2:
            rect = patches.Rectangle((top_bb[i][3], top_bb[i][1]), top_bb[i][2]-top_bb[i][1], top_bb[i][4]-top_bb[i][3], linewidth=2.5, edgecolor=color[1], facecolor='none')
            #ax = plt.gca()
            ax.add_patch(rect)
            
        if shape:
            title = f'{i}: {img.shape[:2]}'
        if simMap[0]:                    
            #title = f'({i}, {round(simMap[1][0][i].cpu().item(), 3)}, {round(simMap[2][i].item(), 3)})' #
            title = f'({i}, {round(simMap[2][i].item(), 3)})'  
            pass
            
        ax.set_title(title)
        
        if proto_title:
            if i % 4 ==0:
                ax.set_ylabel(dic_title[i])
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                ax.axis('off')
        else:
            ax.axis('off')
            
#####################
####################
def plot_annotations(df, dser_dir, s=5, fs=12, n=20, nrow=2, ncol=5, size=(10,10)):
    fig = plt.subplots(layout='constrained', figsize=size)
    plt.rcParams['font.size'] = f'{fs}'
    
    filenames = df.filename.unique()
    masks = {}
    r = 496/512
    
    if not nrow:
        nrow = math.ceil(n/ncol)
    
    for idx, fname in enumerate(filenames):
        if idx >= n:
            break
            
        plt.subplot(nrow, ncol, idx+1)
        ax = plt.gca()
        
        tmp_df = df[df.filename==fname].reset_index(drop=True)
    
        img_path = os.path.join(dser_dir, f'{fname}.png')
        img = plt.imread(img_path)
        ax.imshow(img)
        ax.set_title(fname)
        
        mask = np.zeros((496, 496))
        for j in range(len(tmp_df)):
            row = tmp_df.iloc[j]
            x, y = int(row['x']*r), int(row['y']*r)
            mask[x,y] = 1
            ax.scatter(x, y, c='red', s=s)
        ax.axis('off')
        masks[fname] = mask
    return masks



def plot_inference_heatmap(model, imgs, proto_num=9, n=3, fs=12, ncol=6, vmax=0.002, vmax_=None, size =(16,5)):
    
    plt.rcParams['font.size'] = f'{fs}'
    fig = plt.figure(figsize=size, layout='constrained')
    
    u = 0
    for i in range(n):
        images = torch.from_numpy(imgs[i][1])
        X = imgs[i][0].to('cuda')   
        
        distances = model.prototype_distances(X)
        log_sim_map = model.distance_2_similarity(distances).cpu().numpy()
        
        proto_heat = get_HD_similarity_map_v2(images, log_sim_map, yhat=1) 
        heatmap = proto_heat[proto_num]
        
        u += 1        
        ax = fig.add_subplot(1, ncol, u)
        ax.imshow(imgs[i][1][0])
        ax.axis('off')
        
        u += 1 
        if u < 3:
            vmax = vmax if vmax_ else heatmap['vmax']
        ax = fig.add_subplot(1, ncol, u)    
        img = ax.imshow(heatmap['heatmap'], extent=heatmap['extent'], interpolation='none', cmap=heatmap['cmap'], vmin=-vmax, vmax=vmax)
        ax.imshow(heatmap['overlay'], extent=heatmap['extent'], interpolation='none', cmap=heatmap['cmap_original'], alpha=.25)   
        ax.axis('off')
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(img, cax=cax)