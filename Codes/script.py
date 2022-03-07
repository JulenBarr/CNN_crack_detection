import statistics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
from random import shuffle, randint, sample
import scipy.io as sio
from time import time
# from IPython.display import Image
from PIL import Image
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import regularizers
import os
import script as ut

from sklearn.preprocessing import MaxAbsScaler
# Fix seeds of random generators to make this notebook's output stable across runs
np.random.seed(126)
tf.random.set_seed(126)
random.seed(126)

# Check available CPUs and GPUs
import multiprocessing
print("Number of CPUs Available: ", multiprocessing.cpu_count())
print("Number of GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("No GPU version of TF")

# To plot pretty figures
# %matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# Visualization
import napari
# %gui qt5

################################
# Where to save figures
# PROJECT_ROOT_DIR = "."
# CHAPTER_ID = "cnn"
# IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
# os.makedirs(IMAGES_PATH, exist_ok=True)

# def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
#     path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
#     print("Saving figure", fig_id)
#     if tight_layout:
#         plt.tight_layout()
#     plt.savefig(path, format=fig_extension, dpi=resolution)
################################

ratio = 5.0
s = 5.0
def read_image(pathlib):
    '''
    This function gets a path to a single image (no volume) of any channel and
    format and returns a numpy array.

    params:
    pathlib: pathlib path to the image

    return:
    img: numpy array with pixel values.
    '''
    with Image.open(pathlib) as data:
        img = np.array(data)
    return img

def get_shuffled(images,labels):
    """
        This function returns shuffled images and labels.
        
        params: 
            images: list, given images to be shuffled
            labels: list, given labels to be shuffled
        
        return:
            image_shuffled: list, shuffled images
            labels_shuffled: list, shuffled labels
    """
    num_img = len(images) # number of images
    range_num = list(range(num_img))
    for i in range(5):
        shuffle(range_num)

    image_shuffled, labels_shuffled = [],[]
    for i in range_num:
        image_shuffled.append(images[i])
        labels_shuffled.append(labels[i])

    return image_shuffled, labels_shuffled


def get_patch(image, seg_mask, flag_shuffle=True, ratio=ratio , h=13, s=s):
    """
        This function returns patches extracted from the given image and 
        corresponding structured labels extracted from the ground truth (image label). 
        The ratio of the number of negative patches to the number of 
        positive patches is prescribed by "ratio".
        
        params:
            image: np.array, given image
            seg_mask: np.array, ground truth (label) corresponding to the given image
            flag_shuffle: bool, indicates whether extracted patches and labels are shuffled
            ratio: float, the ratio of the number of negative patches to 
                    the number of positive patches
            h: int, input patch size = [2h+1, 2h+1], with the pixel at the center
            s: int, output patch size = [2s+1, 2s+1], with the pixel at the center
            
        return:
            patches: list, a list of patches with size [2h+1, 2h+1, 3] extracted from the given image
            labels: list, a list of target structured labels with size [(2s+1)^2,] extracted from the given ground truth
    """
    
    indices_pos = np.where(seg_mask == 1) # indices of positive pixels
    indices_neg = np.where(seg_mask == 0) # indices of negative pixels
    patches, labels = [], [] # for all patches and labels extracted from the given image
    
    ## Padding      
    img_paddings = tf.constant([[h,h], [h,h], [0,0]])
    img_tensor = tf.constant(image)
    image_pad = tf.pad(img_tensor, img_paddings, 'SYMMETRIC').numpy() # symmetric paddings to images
    
    label_paddings = tf.constant([[s,s], [s,s]])
    seg_mask = tf.constant(seg_mask)
    seg_mask_pad = tf.pad(seg_mask, label_paddings, 'CONSTANT').numpy() # zero paddings to labels
    
    # Positive input patches and output patches (structured labels)
    count, length = 0, len(indices_pos[0])
    for i,ind in enumerate(indices_pos[0]):
        x,y = ind, indices_pos[1][i]
        patches.append(image_pad[x:x+2*h+1, y:y+2*h+1])
        labels.append(seg_mask_pad[x:x+2*s+1, y:y+2*s+1].flatten())
        count+=1
            
    # Negative input patches and output patches (structured labels)
    no_negs = int(ratio*len(patches)) # number of negative patches
    count, length = 0, len(indices_neg[0])
    while count < no_negs:
        rand_int = randint(0, length)-1 # random index
        x, y = indices_neg[0][rand_int], indices_neg[1][rand_int]
        patches.append(image_pad[x:x+2*h+1, y:y+2*h+1])
        labels.append(seg_mask_pad[x:x+2*s+1, y:y+2*s+1].flatten())
        count+=1
        
    # Shuffle all patches and labels
    if flag_shuffle: 
        patches, labels = get_shuffled(patches, labels)
    
    return patches, labels


def load_data(image_path, label_path, file_id, flag_shuffle=True, ratio=ratio, h=13, s=s):
    """
        This function returns arrays that contain all patches
        extracted from images and corresponding labels extracted 
        from ground truth of all images with indices listed in "file_id"
        
        params:
            image_path: str, path to images
            label_path: str, path to ground truth (labels of images)
            file_id: list, a list of indices of images
            flag_shuffle: bool, indicates whether extracted patches and labels are shuffled
            ratio: float, the ratio of the number of negative patches to 
                    the number of positive patches
            h: int, input patch size = [2h+1, 2h+1], with the pixel at the center
            s: int, output patch size = [2s+1, 2s+1], with the pixel at the center
        
        return:
            all_patches: np.array, [num_patches, 2h+1, 2h+1, 3], all patches extracted from images with indices in file_id
            all_labels: np.array, [num_patches, (2s+1)^2], all structured label extracted from images with indices in file_id
    """
    
    label_files = os.listdir(label_path) # list of names of label files
    label_files_remained = [label_files[i] for i in file_id] # consider the instances prescribed in file_id   
    
    all_patches = np.empty((0,2*h+1,2*h+1,3))
    all_labels = np.empty((0,(2*s+1)**2))

    for label_file in label_files_remained: # loop over label_files
        image_file = label_file.split('.')[0]+'.tif' # image file name
        img = read_image(image_path / image_file) # load image data
        #img = mpimg.imread(os.path.join(image_path, image_file))
        #img = 2*(img - img.min()) / (img.max()-img.min()) -1 #Scale to [-1,1]
        img = 2*((img - img.min()) / (img.max() - img.min())) - 1
        
        seg_mask = mpimg.imread(os.path.join(label_path, label_file)) # load corresponding label
        seg_mask[seg_mask == 1] = 0  #in case is not 1,0 binary
        seg_mask[seg_mask > 1] = 1
        
        # preprocess data to get input input patches and output patches (structured labels)
        patches, labels = get_patch(img, seg_mask, flag_shuffle=flag_shuffle, ratio=ratio, h=h, s=s) #AQUI HE CAMBIADO EL RATIO
        all_patches = np.concatenate((all_patches, patches), axis = 0)
        all_labels = np.concatenate((all_labels, labels), axis = 0)
        
    
    return all_patches, all_labels

def plot_samples(image_path, label_path, sample_id):
    label_files = os.listdir(label_path)
    label_files_test = [label_files[i] for i in sample_id[:6]]

    fig = plt.figure(figsize=(18,8))
    for i, label_file in enumerate(label_files_test):
        image_file = label_file.split('.')[0] + '.tif' # image file name
        img = read_image(image_path / image_file)/ 255.0 # load image data
        seg_mask = read_image(label_path / label_file) # load corresponding label
        # seg_mask[seg_mask == 1] = 0
        # seg_mask[seg_mask > 1] = 1

        label_temp = np.zeros_like(img, dtype='int32')
        for j in range(3):
            label_temp[:,:,j] = seg_mask
        img_label = np.hstack([img, label_temp])

        ax = fig.add_subplot(3,3,i+1)
        ax.imshow(img_label)
        ax.axis('off')
        ax.set_title('Sample {}'.format(label_file.split('.')[0]), fontsize=20)
    plt.tight_layout()
    plt.show()
    return img_label
    # plt.savefig('images\\train_samples.png')
    
    
def get_batches(image, h=13):
    """
        This function extract patches from the given image with 
        the patch size being [2h+1, 2h+1].
        
        params:
            image, np.arrays, given image
            h: int, input patch size: [2h+1, 2h+1], with the pixel at the center
            
        return:
            patches: np.arrays, extracted patches from the given image
    """
    m,n,p = image.shape
    patches = []
    for i in range(0,m):
        for j in range(0,n):
            if i-h-1 < 0 or j-h-1 < 0 or i+h+1 > m or j+h+1 > n:
                continue
            else:
                patches.append(image[i-h:i+h+1, j-h:j+h+1])
    return np.array(patches)


def infer(prediction, m, n, h=13, s=s):
    """
        This function sums up the structured outputs from s_out^2 units of 
        all pixels in an image. Note that the structured outputs of pixels
        can overlap.
        
        params:
            prediction: [num_pixels, num_output_units]
                        predictions from all s_out^2 output units of each pixel
            m: number of pixels in horizontal direction of original image
            n: number of pixels in vertical direction of original image
            h: int, input patch size: [2h+1, 2h+1], with the pixel at the center
            s: int, output patch size: [2s+1, 2s+1], with the pixel at the center
            
        return:
            array of image's original size that contains aggregate predictions
    """
    out = np.zeros((m,n), np.float32)
    count = 0
    for i in range(0,m):
        for j in range(0,n):
            if i-h-1 < 0 or j-h-1 < 0 or i+h+1 > m or j+h+1 > n:
                continue
            else:
                out[i-s:i+s+1,j-s:j+s+1] += prediction[count].reshape((2*s+1,2*s+1))
                count += 1
    return out[h:m-h,h:n-h]

def model(h=13,s=s):
    """
        This function creates a CNN model.
        
        params: 
            h: int, input patch: [2h+1, 2h+1], with the pixel at the center
            s: int, output patch: [2s+1, 2s+1], with the pixel at the center
        
        return:
            model: tf.keras.sequential
    """
    model = keras.models.Sequential()
    initializer = tf.keras.initializers.GlorotNormal()
    # 1st Convolutional Layer: 16 feature maps, kernel size: 3x3, stride: 1, padding: zeros
    model.add(keras.layers.Conv2D(16, (3,3), 
                                    activity_regularizer=regularizers.l2(0.00025),
                                    kernel_initializer=initializer,
                                    strides=(1, 1), padding='same', activation='relu',
                                    input_shape=[2*h+1,2*h+1,3]))
    
    # 2nd Convolutional Layer: 16 feature maps, kernel size: 3x3, stride: 1, padding: zeros
    model.add(keras.layers.Conv2D(16, (3,3),
                                        activity_regularizer = regularizers.l2(0.00025),
                                        kernel_initializer=initializer,
                                        strides=(1, 1), padding='same', activation='relu'))
    
    # Max Pooling Layer: window size: 2x2, stride: 2
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2))) #Faltaban los padding 
    
    # 3rd Convolutional Layer: 32 feature maps, kernel size: 3x3, stride: 1, padding: zeros
    model.add(keras.layers.Conv2D(32, (3,3), 
                                        activity_regularizer = regularizers.l2(0.00025),
                                        kernel_initializer = initializer,
                                        strides=(1, 1), padding='same', activation='relu'))  
    # 4th Convolutional Layer: 32 feature maps, kernel size: 3x3, stride: 1, padding: zeros
    model.add(keras.layers.Conv2D(32, (3,3), 
                                            activity_regularizer = regularizers.l2(0.00025),
                                            kernel_initializer=initializer,
                                            strides=(1, 1), padding='same', activation='relu'))
    # Max Pooling Layer: window size: 2x2, stride: 2
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    # Fully Connected Layers
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64,kernel_initializer=initializer,
                                    activity_regularizer =regularizers.l2(0.00025),
                                    activation='relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(64,kernel_initializer=initializer,
                                    activity_regularizer =regularizers.l2(0.00025),
                                    activation='relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense((2*s+1)**2, kernel_initializer=initializer,
                                activity_regularizer=regularizers.l2(0.00025),
                                activation='tanh'))
    return model

def save_output(img, image_file, label, out_norm, save_path):
    """
        This function plots and saves figures: 
        original image, label, crack prediction image
        
        params:
            img: np.arrays, original image
            image_file: str, file name of the given image
            label: np.arrays, label of the original image
            out_norm: crack classification results
            save_path: path to save prediction images
    """
    out_channels = np.zeros_like(img)
    label_temp = np.zeros_like(img, dtype='int32')
    for i in range(3):
        out_channels[:,:,i] = out_norm
        label_temp[:,:,i] = label
        
    white_space = np.ones((img.shape[0],10,3), dtype='int32')
    save_image = np.hstack([img, white_space, label_temp, white_space, out_channels]) # stack arrays in sequence horizontally
    plt.figure(figsize=(12,6))
    plt.imshow(save_image)
    plt.axis("off")
    plt.savefig(os.path.join(save_path, image_file.replace('jpg','png')))
    plt.show()
    
    
def calc_precision_recall(ytrue, ypred):
    """
        This function calculates precision, recall, and F1 scores.
        
        params:
            ytrue: np.arrays, ground truth
            ypred: np.arrays, model prediction
        
        return:
            precision: float
            recall: float
            F1: float
    """
    true_pos = len(np.nonzero(np.logical_and(ytrue==ypred, ytrue==1))[0])
    true_neg = len(np.nonzero(np.logical_and(ytrue==ypred, ytrue==0))[0])
    false_pos = len(np.nonzero(np.logical_and(ytrue!=ypred, ytrue==0))[0])
    false_neg = len(np.nonzero(np.logical_and(ytrue!=ypred, ytrue==1))[0])
    
    precision = float(true_pos) / (true_pos+false_pos)
    recall = float(true_pos) / (true_pos+false_neg)
    F1 = float(2*precision*recall) / (precision+recall)
    
    return precision, recall, F1


def crack_pred(model, label_path, image_path, save_path, test_id, h=13, s=s, prob=0.5):
    """
        This function uses the trained CNN model to perform crack prediction on testing samples.
        It plots and saves figures: original image + ground truth + crack prediction image
        
        params:
            model: tf.keras.sequential, the trained CNN model
            label_path: str, path to the ground truth
            image_path: str, path to the images
            save_path: str, path to save prediction images
            test_id: list, indices of testing instances
            h: int, input patch size: [2h+1, 2h+1]
            s: int, output patch size: [2s+1, 2s+1]
            
        return:
            precision: float, precision score
            recall: float, recall score
            F1: float, F1 score
    """
    label_files = os.listdir(label_path) # list of names of label files
    
    label_files_remained = [label_files[i] for i in test_id] # consider the instances prescribed in file_id
    paddings = tf.constant([[h,h], [h,h], [0,0]])

    tot_p, tot_r, tot_f = 0, 0, 0
    count = 0
    l_out_norm = []
    for label_file in label_files_remained:
        # Load the image of Test instance and norm.
        image_file = label_file.split('.')[0]+'.tif' # image file name
        #img = read_image(image_path / image_file)/ 255.0 # load image data. FORMA ORIGINAL
        img = mpimg.imread(os.path.join(image_path, image_file))
        #img = 2*(img - img.min()) / (img.max()-img.min()) -1 #Scale to [-1,1]
        img = 2*((img - img.min()) / (img.max() - img.min())) - 1

        # Symmetric padding
        img_tensor = tf.constant(img)
        img1 = tf.pad(img_tensor, paddings, 'symmetric').numpy()
        m,n,p = img1.shape
        X_test = get_batches(img1, h=h) # patches extracted from an image

        # Load the label of Test instance
        seg_mask = read_image(label_path / label_file)
        seg_mask[seg_mask == 1] = 0
        seg_mask[seg_mask > 1] = 1

        # Prediciton Results of Test instance
        total_prediction = model.predict(X_test)
        
        # Probability Map
        out = infer(total_prediction, m, n, h=h, s=s)
        out_norm = (out - out.min()) / out.max() # normalized to [0,1]]

        # Crack prediction: decision threshold = prob = 0.5 default
        # if predicted probability >= prob, it is a Crack pixel
        # if predicted probability < prob, it is a Non-Crack pixel
        pred_mask = out_norm.copy()
        pred_mask[pred_mask >= prob] = 1
        pred_mask[pred_mask < prob] = 0
        l_out_norm.append(out_norm)

        # Save figures
        save_output(img, image_file, seg_mask, pred_mask, save_path)
        precision, recall, F1 = calc_precision_recall(seg_mask, pred_mask)
        tot_p += precision 
        tot_r += recall
        tot_f += F1
        count += 1
    precision = tot_p/count
    recall = tot_r/count
    f1 = tot_f/count
    
    return precision, recall, f1,l_out_norm


