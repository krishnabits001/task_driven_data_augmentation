import numpy as np
#from sklearn.metrics import f1_score
import nibabel as nib
import os

#to make directories
import pathlib
from skimage import transform

class dataloaderObj:

    #define functions to load data from ACDC dataset
    def __init__(self,cfg):
        #print('dataloaders init')
        self.data_path_tr=cfg.data_path_tr
        self.data_path_tr_cropped=cfg.data_path_tr_cropped
        #self.slic_path_tr_cropped=cfg.slic_path_tr_cropped
        self.target_resolution=cfg.target_resolution
        self.size=cfg.size
        self.num_classes=cfg.num_classes

    def normalize_minmax_data(self, image_data):
        """
        # 3D MRI scan is normalized to range between 0 and 1 using min-max normalization.
        Here, the minimum and maximum values are used as 2nd and 98th percentiles respectively from the 3D MRI scan.
        We expect the outliers to be away from the range of [0,1].
        input params :
            image_data : 3D MRI scan to be normalized using min-max normalization
        returns:
            final_image_data : Normalized 3D MRI scan obtained via min-max normalization.
        """
        min_val_2p=np.percentile(image_data,2)
        max_val_98p=np.percentile(image_data,98)
        final_image_data=np.zeros((image_data.shape[0],image_data.shape[1],image_data.shape[2]), dtype=np.float64)
        # min-max norm on total 3D volume
        final_image_data=(image_data-min_val_2p)/(max_val_98p-min_val_2p)
        return final_image_data


    def load_acdc_imgs(self, study_id_list,ret_affine=0):
        """
        #Load ACDC data image and its label with pixel dimensions
        input params :
            study_id_list: id no of the image to be loaded
            ret_affine: variable to enable returning of affine transformation matrix of the loaded image
        returns :
            image_data_test_sys : normalized 3D image
            label_data_test_sys : 3D label mask of the image
            pixel_size : pixel dimensions of the loaded image
            affine_tst : affine transformation matrix of the loaded image
        """

        for study_id in study_id_list:
            path_files=str(self.data_path_tr)+str(study_id)+'/'
            systole_lstfiles = []  # create an empty list
            for dirName, subdirList, fileList in os.walk(path_files):
                    fileList.sort()
                    for filename in fileList:
                        if "_frame01" in filename.lower():
                            systole_lstfiles.append(os.path.join(dirName,filename))
                        elif "_frame04" in filename.lower():
                            systole_lstfiles.append(os.path.join(dirName,filename))

        # Load the 3D image
        image_data_test_load = nib.load(systole_lstfiles[0])
        image_data_test_sys=image_data_test_load.get_data()
        pixel_size=image_data_test_load.header['pixdim'][1:4]
        affine_tst=image_data_test_load.affine

        # Normalize input data
        image_data_test_sys=self.normalize_minmax_data(image_data_test_sys)

        # Load the segmentation mask
        label_data_test_load = nib.load(systole_lstfiles[1])
        label_data_test_sys=label_data_test_load.get_data()

        if(ret_affine==0):
            return image_data_test_sys,label_data_test_sys,pixel_size
        else:
            return image_data_test_sys,label_data_test_sys,pixel_size,affine_tst


    def crop_or_pad_slice_to_size_1hot(self, img_slice, nx, ny):

        """
        To crop the input 2D slice for the given dimensions in 1-hot encoding format)
        input params :
            image_slice : 2D slice to be cropped (in 1-hot encoding format)
            nx : dimension in x
            ny : dimension in y
        returns:
            slice_cropped : cropped 2D slice
        """
        slice_cropped=np.zeros((nx,ny,self.num_classes))
        x, y, _ = img_slice.shape

        x_s = (x - nx) // 2
        y_s = (y - ny) // 2
        x_c = (nx - x) // 2
        y_c = (ny - y) // 2

        if x > nx and y > ny:
            slice_cropped = img_slice[x_s:x_s + nx, y_s:y_s + ny]
        else:
            slice_cropped = np.zeros((nx, ny,self.num_classes))
            if x <= nx and y > ny:
                slice_cropped[x_c:x_c + x, :] = img_slice[:, y_s:y_s + ny]
            elif x > nx and y <= ny:
                slice_cropped[:, y_c:y_c + y] = img_slice[x_s:x_s + nx, :]
            else:
                slice_cropped[x_c:x_c + x, y_c:y_c + y] = img_slice[:, :]

        return slice_cropped

    def crop_or_pad_slice_to_size(self, img_slice, nx, ny):
        """
        To crop the input 2D slice for the given dimensions
        input params :
            image_slice : 2D slice to be cropped
            nx : dimension in x
            ny : dimension in y
        returns:
            slice_cropped : cropped 2D slice
        """
        slice_cropped=np.zeros((nx,ny))
        x, y = img_slice.shape

        x_s = (x - nx) // 2
        y_s = (y - ny) // 2
        x_c = (nx - x) // 2
        y_c = (ny - y) // 2

        if x > nx and y > ny:
            slice_cropped = img_slice[x_s:x_s + nx, y_s:y_s + ny]
        else:
            slice_cropped = np.zeros((nx, ny))
            if x <= nx and y > ny:
                slice_cropped[x_c:x_c + x, :] = img_slice[:, y_s:y_s + ny]
            elif x > nx and y <= ny:
                slice_cropped[:, y_c:y_c + y] = img_slice[x_s:x_s + nx, :]
            else:
                slice_cropped[x_c:x_c + x, y_c:y_c + y] = img_slice[:, :]

        return slice_cropped

    def preprocess_data(self, img, mask, pixel_size,label_present=1):
        """
        To preprocess the input 3D volume into given target resolution and crop them into dimensions specified in the init_acdc.py file
        input params :
            img : input 3D image volume to be processed
            mask : corresponding 3D segmentation mask to be processed
            pixel_size : the native pixel size of the input image
            label_present : to indicate if the image has labels provided or not (used for unlabeled images)
        returns:
            cropped_img : processed and cropped 3D image
            cropped_mask : processed and cropped 3D segmentation mask
        """
        nx,ny=self.size

        #scale vector to rescale to the target resolution
        scale_vector = [pixel_size[0] / self.target_resolution[0], pixel_size[1] / self.target_resolution[1]]

        for slice_no in range(img.shape[2]):

            slice_img = np.squeeze(img[:, :, slice_no])
            slice_rescaled = transform.rescale(slice_img,
                                               scale_vector,
                                               order=1,
                                               preserve_range=True,
                                               mode = 'constant')
            if(label_present==1):
                slice_mask = np.squeeze(mask[:, :, slice_no])
                mask_rescaled = transform.rescale(slice_mask,
                                              scale_vector,
                                              order=0,
                                              preserve_range=True,
                                              mode='constant')

            slice_cropped = self.crop_or_pad_slice_to_size(slice_rescaled, nx, ny)
            if(label_present==1):
                mask_cropped = self.crop_or_pad_slice_to_size(mask_rescaled, nx, ny)

            if(slice_no==0):
                cropped_img=np.reshape(slice_cropped,(nx,ny,1))
                if(label_present==1):
                    cropped_mask=np.reshape(mask_cropped,(nx,ny,1))
            else:
                slice_cropped_tmp=np.reshape(slice_cropped,(nx,ny,1))
                cropped_img=np.concatenate((cropped_img,slice_cropped_tmp),axis=2)
                if(label_present==1):
                     mask_cropped_tmp=np.reshape(mask_cropped,(nx,ny,1))
                     cropped_mask=np.concatenate((cropped_mask,mask_cropped_tmp),axis=2)

        if(label_present==1):
            return cropped_img,cropped_mask
        else:
            return cropped_img

    def load_acdc_cropped_img_labels(self, train_ids_list,label_present=1):
        """
        # Load the already created and stored a-priori acdc data and its labels that are preprocessed and cropped to given dimensions
        input params :
            train_ids_list : patient ids of the image and label pairs to be loaded
            label_present : to indicate if the image has labels provided or not (used for unlabeled images)
        returns:
            img_cat : stack of 3D images of all the patient id nos.
            mask_cat : corresponding stack of 3D segmentation masks of all the patient id nos.
        """

        count=0
        for study_id in train_ids_list:
            #print("study_id",study_id)
            img_fname = str(self.data_path_tr_cropped)+str(study_id)+'/img_cropped.npy'
            img_tmp=np.load(img_fname)
            if(label_present==1):
                mask_fname = str(self.data_path_tr_cropped)+str(study_id)+'/mask_cropped.npy'
                mask_tmp=np.load(mask_fname)

            if(count==0):
                img_cat=img_tmp
                if(label_present==1):
                    mask_cat=mask_tmp
                count=1
            else:
                img_cat=np.concatenate((img_cat,img_tmp),axis=2)
                if(label_present==1):
                    mask_cat=np.concatenate((mask_cat,mask_tmp),axis=2)
        if(label_present==1):
            return img_cat,mask_cat
        else:
            return img_cat
