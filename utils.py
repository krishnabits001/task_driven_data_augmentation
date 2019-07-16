import numpy as np
import scipy.ndimage.interpolation

from skimage import transform
import random

import os
import re


def augmentation_function(ip_list, dt, labels_present=1, en_1hot=0):
    '''
    To generate affine augmented image,label pairs.

    ip params:
        ip_list: list of 2D slices of images and its labels if labels are present
        dt: dataloader object
        labels_present: to indicate if labels are present or not
        en_1hot: to indicate labels are used in 1-hot encoding format
    returns:
        sampled_image_batch : augmented images generated
        sampled_label_batch : corresponding augmented labels
    '''

    if(len(ip_list)==2 and labels_present==1):
        images = ip_list[0]
        labels = ip_list[1]
    else:
        images=ip_list[0]

    if images.ndim > 4:
        raise AssertionError('Augmentation will only work with 2D images')

    new_images = []
    new_labels = []
    num_images = images.shape[0]

    for index in range(num_images):

        img = np.squeeze(images[index,...])
        if(labels_present==1):
            lbl = np.squeeze(labels[index,...])

        do_rotations,do_scaleaug,do_fliplr,do_simple_rot=0,0,0,0
        #option 5 is to not perform any augmentation i.e, use the original image
        aug_select = np.random.randint(5)

        if(np.max(img)>0.001):
            if(aug_select==0):
                do_rotations=1
            elif(aug_select==1):
                do_scaleaug=1
            elif(aug_select==2):
                do_fliplr=1
            elif(aug_select==3):
                do_simple_rot=1

        # ROTATE between angle -15 to 15
        if do_rotations:
            angles = [-15,15]
            random_angle = np.random.uniform(angles[0], angles[1])
            img = scipy.ndimage.interpolation.rotate(img, reshape=False, angle=random_angle, axes=(1, 0),order=1)
            if(labels_present==1):
                if(en_1hot==1):
                    lbl = scipy.ndimage.interpolation.rotate(lbl, reshape=False, angle=random_angle, axes=(1, 0),order=1)
                else:
                    lbl = scipy.ndimage.interpolation.rotate(lbl, reshape=False, angle=random_angle, axes=(1, 0),order=0)

        # RANDOM SCALE
        if do_scaleaug:
            n_x, n_y = img.shape
            #scale factor between 0.9 and 1.1
            scale_fact_min=0.9
            scale_fact_max=1.1
            scale_val = round(random.uniform(scale_fact_min,scale_fact_max), 2)
            slice_rescaled = transform.rescale(img, scale_val, order=1, preserve_range=True, mode = 'constant')
            img = dt.crop_or_pad_slice_to_size(slice_rescaled, n_x, n_y)
            if(labels_present==1):
                if(en_1hot==1):
                    slice_rescaled = transform.rescale(lbl, scale_val, order=1, preserve_range=True, mode = 'constant')
                    lbl = dt.crop_or_pad_slice_to_size_1hot(slice_rescaled, n_x, n_y)
                else:
                    slice_rescaled = transform.rescale(lbl, scale_val, order=0, preserve_range=True, mode = 'constant')
                    lbl = dt.crop_or_pad_slice_to_size(slice_rescaled, n_x, n_y)

        # RANDOM FLIP
        if do_fliplr:
            coin_flip = np.random.randint(2)
            if coin_flip == 0:
                img = np.fliplr(img)
                if(labels_present==1):
                    lbl = np.fliplr(lbl)

        # Simple rotations at angles of 45 degrees
        if do_simple_rot:
            fixed_angle = 45
            random_angle = np.random.randint(8)*fixed_angle

            img = scipy.ndimage.interpolation.rotate(img, reshape=False, angle=random_angle, axes=(1, 0),order=1)
            if(labels_present==1):
                if(en_1hot==1):
                    lbl = scipy.ndimage.interpolation.rotate(lbl, reshape=False, angle=random_angle, axes=(1, 0),order=1)
                else:
                    lbl = scipy.ndimage.interpolation.rotate(lbl, reshape=False, angle=random_angle, axes=(1, 0),order=0)

        new_images.append(img[..., np.newaxis])
        if(labels_present==1):
            new_labels.append(lbl[...])

    sampled_image_batch = np.asarray(new_images)
    if(labels_present==1):
        sampled_label_batch = np.asarray(new_labels)

    if(len(ip_list)==2 and labels_present==1):
        return sampled_image_batch, sampled_label_batch
    else:
        return sampled_image_batch

def calc_deform(cfg, mu=0,sigma=10, order=3):
    '''
    To generate a batch of smooth deformation fields for the specified mean and standard deviation value.

    input params:
        cfg: experiment config parameter (contains image dimensions, batch_size, etc)
        mu: mean value for the normal distribution
        sigma: standard deviation value for the normal distribution
        order: order of interpolation; 3 = bicubic interpolation
    returns:
        flow_vec: batch of deformation fields generated
    '''

    flow_vec = np.zeros((cfg.batch_size,cfg.img_size_x,cfg.img_size_y,2))

    for i in range(cfg.batch_size):
        #mu, sigma = 0, 10 # mean and standard deviation
        dx = np.random.normal(mu, sigma, 9)
        dx_mat = np.reshape(dx,(3,3))
        dx_img = transform.resize(dx_mat, output_shape=(cfg.img_size_x,cfg.img_size_y), order=order,mode='reflect')

        dy = np.random.normal(mu, sigma, 9)
        dy_mat = np.reshape(dy,(3,3))
        dy_img = transform.resize(dy_mat, output_shape=(cfg.img_size_x,cfg.img_size_y), order=order,mode='reflect')


        flow_vec[i,:,:,0] = dx_img
        flow_vec[i,:,:,1] = dy_img

    return flow_vec

def shuffle_minibatch(ip_list, batch_size=20,num_channels=1,labels_present=1,axis=2):
    '''
    To sample a minibatch images of batch_size from all the available 3D volume of images.

    input params:
        ip_list: list of 2D slices of images and its labels if labels are present
        batch_size: number of 2D slices to consider for the training
        labels_present: to indicate labels are used in 1-hot encoding format
        num_channels : no of channels of the input image
        axis : the axis along which we want to sample the minibatch -> axis vals : 0 - for sagittal, 1 - for coronal, 2 - for axial
    returns:
        image_data_train_batch: concatenated 2D slices randomly chosen from the total input data
        label_data_train_batch: concatenated 2D slices of labels with indices corresponding to the input data selected.
    '''

    if(len(ip_list)==2 and labels_present==1):
        image_data_train = ip_list[0]
        label_data_train = ip_list[1]
    else:
        image_data_train=ip_list[0]

    img_size_x=image_data_train.shape[0]
    img_size_y=image_data_train.shape[1]
    img_size_z=image_data_train.shape[2]

    len_of_train_data=np.arange(image_data_train.shape[axis])

    randomize=np.random.choice(len_of_train_data,size=len(len_of_train_data),replace=True)

    count=0
    for index_no in randomize:
        if(axis==2):
            img_train_tmp=np.reshape(image_data_train[:,:,index_no],(1,img_size_x,img_size_y,num_channels))
            if(labels_present==1):
                label_train_tmp=np.reshape(label_data_train[:,:,index_no],(1,img_size_x,img_size_y))
        elif(axis==1):
            img_train_tmp=np.reshape(image_data_train[:,index_no,:,],(1,img_size_x,img_size_z,num_channels))
            if(labels_present==1):
                label_train_tmp=np.reshape(label_data_train[:,index_no,:],(1,img_size_x,img_size_z))
        else:
            img_train_tmp=np.reshape(image_data_train[index_no,:,:],(1,img_size_y,img_size_z,num_channels))
            if(labels_present==1):
                label_train_tmp=np.reshape(label_data_train[index_no,:,:],(1,img_size_y,img_size_z))

        if(count==0):
            image_data_train_batch=img_train_tmp
            if(labels_present==1):
                label_data_train_batch=label_train_tmp
        else:
            image_data_train_batch=np.concatenate((image_data_train_batch, img_train_tmp),axis=0)
            if(labels_present==1):
                label_data_train_batch=np.concatenate((label_data_train_batch, label_train_tmp),axis=0)
        count=count+1
        if(count==batch_size):
            break

    if(len(ip_list)==2 and labels_present==1):
        return image_data_train_batch, label_data_train_batch
    else:
        return image_data_train_batch

def change_axis_img(ip_list, labels_present=1, def_axis_no=2, cat_axis=0):
    '''
    To swap the axes of 3D volumes as per the network input
    input params:
        ip_list: list of 2D slices of images and its labels if labels are present
        labels_present: to indicate labels are used in 1-hot encoding format
        def_axis_no: axis which needs to be swapped (default axial direction here)
        cat_axis: axis along which the images need to concatenated
    returns:
        mergedlist_img: swapped axes 3D volumes
        mergedlist_labels: corresponding swapped 3D volumes
    '''
    # Swap axes of 3D volume according to the input of the network
    if(len(ip_list)==2 and labels_present==1):
        labeled_data_imgs = ip_list[0]
        labeled_data_labels = ip_list[1]
    else:
        labeled_data_imgs=ip_list[0]

    #can also define in an init file - base values
    img_size_x=labeled_data_imgs.shape[0]
    img_size_y=labeled_data_imgs.shape[1]

    total_slices = labeled_data_imgs.shape[def_axis_no]
    for slice_no in range(total_slices):

        img_test_slice = np.reshape(labeled_data_imgs[:, :, slice_no], (1, img_size_x, img_size_y, 1))
        if(labels_present==1):
            label_test_slice = np.reshape(labeled_data_labels[:, :, slice_no], (1, img_size_x, img_size_y))

        if (slice_no == 0):
            mergedlist_img = img_test_slice
            if(labels_present==1):
                mergedlist_labels = label_test_slice

        else:
            mergedlist_img = np.concatenate((mergedlist_img, img_test_slice), axis=cat_axis)
            if(labels_present==1):
                mergedlist_labels = np.concatenate((mergedlist_labels, label_test_slice), axis=cat_axis)

    if(len(ip_list)==2 and labels_present==1):
        return mergedlist_img,mergedlist_labels
    else:
        return mergedlist_img

def load_val_imgs(val_list,dt,orig_img_dt):
    '''
    To load validation acdc images and labels,pixel resolution
    input params:
        val_list: list of validation patient ids of acdc data
        dt: dataloader object
        orig_img_dt: dataloader for the image
    returns:
        val_label_orig: returns list of labels without any preprocessing applied
        val_img_re: returns list of images post preprocess steps done
        val_label_re: returns list of labels post preprocess steps done
        pixel_val_list: returns list of pixel resolution values of original images
    '''
    val_label_orig=[]
    val_img_list=[]
    val_label_list=[]
    pixel_val_list=[]

    for val_id in val_list:
        val_id_list=[val_id]
        val_img,val_label,pixel_size_val=orig_img_dt(val_id_list)
        val_cropped_img,val_cropped_mask = dt.preprocess_data(val_img, val_label, pixel_size_val)

        #change axis for quicker computation of dice score
        val_img_re,val_labels_re= change_axis_img([val_cropped_img,val_cropped_mask])

        val_label_orig.append(val_label)
        val_img_list.append(val_img_re)
        val_label_list.append(val_labels_re)
        pixel_val_list.append(pixel_size_val)

    return val_label_orig,val_img_list,val_label_list,pixel_val_list

def get_max_chkpt_file(model_path,min_ep=10):
    '''
    To return the checkpoint file that yielded the best dsc value on val images
    input params:
        model_path: directory of the run where the checkpoint files are stored
        min_ep: variable to ensure that the model selected has higher epoch no. than this no. (here its 10).
    returns:
        fin_chkpt_max: checkpoint file with best dsc value
    '''
    for dirName, subdirList, fileList in os.walk(model_path):
        fileList.sort()
        for filename in fileList:
            if "meta" in filename.lower() and 'best_model' in filename:
                numbers = re.findall('\d+',filename)
                if "_v2" in filename:
                    tmp_ep_no=int(numbers[1])
                else:
                    tmp_ep_no=int(numbers[0])
                if(tmp_ep_no>min_ep):
                    chkpt_max=os.path.join(dirName,filename)
                    min_ep=tmp_ep_no
    fin_chkpt_max = re.sub('\.meta$', '', chkpt_max)
    return fin_chkpt_max

def isNotEmpty(s):
    return bool(s and s.strip())

def mixup_data_gen(x_train,y_train,alpha=0.1):
    '''
    # Generator for mixup data - to linearly combine 2 random image,label pairs from the batch of image,label pairs
    input params:
        x_train: batch of input images
        y_train: batch of input labels
        alpha: alpha value of
    returns:
        x_out: linearly combined resultant image
        y_out: linearly combined resultant label
    '''
    len_x_train = x_train.shape[0]
    x_out=np.zeros_like(x_train)
    y_out=np.zeros_like(y_train)

    for i in range(len_x_train):
        lam = np.random.beta(alpha, alpha)
        rand_idx1 = np.random.choice(len_x_train)
        rand_idx2 = np.random.choice(len_x_train)

        x_out[i] = lam * x_train[rand_idx1] + (1 - lam) * x_train[rand_idx2]
        y_out[i] = lam * y_train[rand_idx1] + (1 - lam) * y_train[rand_idx2]

    return x_out, y_out
