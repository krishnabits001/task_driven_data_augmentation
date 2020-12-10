################################################################
# Variable definitions required for CNN graph
################################################################
#Interpolation type and up scaling factor
interp_val=0 # 0 - bilinear interpolation; 1- nearest neighbour interpolation
################################################################

################################################################
# data dimensions, num of classes and resolution
################################################################
# Image Dimensions
img_size_x = 224
img_size_y = 224
# Images are stored in one-dimensional arrays of this length.
img_size_flat = img_size_x * img_size_y
# Number of colour channels for the images: 1 channel for gray-scale image.
num_channels = 1
# Number of classes : # 0-background, 1-rv, 2-myo, 3-lv
num_classes=4 
size=(img_size_x,img_size_y)
target_resolution=(1.36719,1.36719)
################################################################
# data paths
################################################################
#validation_update_step to save values
val_step_update=10
#base dir of network
base_dir='/usr/bmicnas01/data-biwi-01/krishnch/projects/data_aug_seg/'
#data path tr
data_path_tr='/usr/bmicnas01/data-biwi-01/krishnch/datasets/heart_acdc/acdc_bias_corr/patient'
#cropped imgs data_path
data_path_tr_cropped='/usr/bmicnas01/data-biwi-01/krishnch/datasets/heart_acdc/acdc_bias_corr_cropped/patient'
################################################################

################################################################
#network optimization parameters
################################################################
#enable data augmentation
aug_en=1
#batch_size
batch_size=20
struct_name=['rv','myo','lv']
