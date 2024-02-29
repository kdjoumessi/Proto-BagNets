import os
import math
import json
import pandas as pd
import matplotlib.pyplot as plt

#------------------------
def get_users_dataframe(df):
    ''' 
        process the input raw csv file such that for each annotation within the image the a row containing the image 
        name we have a row containing: the image name, the annotation (label) name, the coordinate, and the color

        input:
            - df: user dataframe (annotation file)
    '''       
    filenames, labels, colors, x, y = [], [], [], [], []
    
    for i in range(len(df)): # for each image
        image_name, coordinates = df['image.name'][i], df['coordinates'][i]
        list_coordinates = json.loads(coordinates) # list of dictionary  {'label': 'MA', 'x': , 'y': , 'col': }
        
        for coordinate in list_coordinates: 
            if not coordinate['label']: 
                # skip if there is no label for the image => the image is considerated healthy from the annotator
                continue
            
            labels.append(coordinate['label'])
            colors.append(coordinate['color'])
            filenames.append(image_name)
            x.append(coordinate['x'])
            y.append(coordinate['y'])
            
    df_tmp = pd.DataFrame({'filename': filenames, 'label': labels, 'x': x, 'y': y, 'color': colors})
            
    return df_tmp

#------------------------
def get_images_with_annotation_v2(df, colors):
    '''
        for a single image, output a dictionary containing for each unique anntotation found the locations
         where they are in the image

        params:
            - df (dataframe): dataframe containing a single image with annotation and locations
            - colors (list): list of unique annotation/color/label
    '''
    
    dic_color = {}
    for color in colors: # for each color -> label
        tmp_df = df[df.color==color].reset_index(drop=True) # get all the similar annotation in that image (df)
        x, y, label = [], [], []
        
        for i in range(len(tmp_df)): # for each of that annotation
            y.append(tmp_df.y[i])    # get the cordinates and the label
            x.append(tmp_df['x'][i])
            label.append(tmp_df.label[i])

        # for each color/label, saved the location where they are in the image 
        dic_color[color] = (x, y, tmp_df.label[i])     
    return dic_color

def get_dic_images_with_annotations(df, labels, dataset):
    ''' 
        for each image, create the corresponding annotations (dico). 
        for each image, create a tuple containing a dictionary of the annotations it contains as well as the 
        corresponding cordinates/location in the image. 
        This will be useful to select which annotation to plot given an image 

        params:
            - df (dataframe): the input dataframe
            - label (list): the list of annotation to consider
            - dataset (dataframe): the dataset containing the input data with the corresponding grade
    '''
    dic_plot = {}
    data = df.copy()
    data = data[data.label.isin(labels)]              # only select some annotations
    image_names = data.filename.unique().tolist()
    
    for img_name in image_names: # for each image
        if img_name in dataset.file.values:  # just to make sure the anotated image is within the initial dataset
            tmp_df = data[data.filename==img_name].reset_index(drop=True) # extract each image with amnotations
            Colors = tmp_df.color.unique().tolist() # get the different annotation/color/label used on it

            # for each image contains the label/color with the corresponding annotation coordinates
            dic_col = get_images_with_annotation_v2(tmp_df, Colors) 
            dic_plot[img_name] = (dic_col, True) # for each image save the annotation (color, location)
    
    #  Just to keep dic_plot the same size with the initial dataset for ploting purpose
    for file in dataset.file.tolist():
        if file not in dic_plot.keys():
            dic_plot[file] = ({}, False)  #file without with empty annotation.

    # sort the result for ploting purpose as well        
    myKeys = list(dic_plot.keys())
    myKeys.sort()
    sorted_dict = {i: dic_plot[i] for i in myKeys}
        
    return sorted_dict


def plot_images_from_dic(plot_dic, dir_name, init_data, user='Anne', layout=True, ncol=5, fs=14, nbre_image=10,
                            marker_size=10, size=(20,20)):
    ''' 
        plot the images with annotations

        params:
            - plot_dic (dic): a dictionary containing for each image the list of annotation per type
            - dir_name (str): the path to the images
            - init_data (df): the dataframe containing the images
            - user (str): 
    '''
    
    j = 0 # image nber
    img_list = []
    bs = math.ceil(len(plot_dic)/ncol) # nber of row
    plt.rcParams['font.size'] = f'{fs}'
    
    if layout:
        fig = plt.figure(figsize=size, layout='constrained', dpi=200) #
    else:
        fig = plt.figure(figsize=size)
    
    for img_name, (dic_col, Bool) in plot_dic.items():
        j += 1

        # load and display the image    
        img_path = os.path.join(dir_name, f'{img_name}.png')
        img = plt.imread(img_path)
        ax = fig.add_subplot(bs, ncol, j)
        ax.imshow(img)
        #if j==1:  # just to check x and y axis
        #    ax.scatter(200, 400, s=50)
        
        # load and plot the annotation on top of the image
        if Bool: # if the image contains annotation
            for col, (x, y, label) in dic_col.items():
                ax.scatter(x, y, c=col, s=marker_size)
            
        level = init_data[init_data.filename==f'{img_name}.png']['level'].tolist()[0]
        plt.title(f'{(img_name, level)}')
        #ax.axis('off')
        
        if not layout:
            fig.subplots_adjust(wspace=None, hspace=None) #plt.subplots_adjust(wspace=0.04, hspace=0.13) (0,0)
            plt.tight_layout()
        
        img_list.append(img_name)
        if nbre_image:            
            if j==nbre_image:
                break 

    print(img_list)
    return img_list