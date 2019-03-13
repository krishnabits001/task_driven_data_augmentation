# task_driven_data_augmentation

The code is for the article "Semi-Supervised and Task-Driven Data Augmentation" which got accepted as an ORAL presentation at IPMI 2019 (26th international conference on Information Processing in Medical Imaging).
The method yields competitive segmentation performance with just 1 labelled training volume.

Authors:

Krishna Chaitanya (email : krishna.chaitanya@vision.ee.ethz.ch),

Christian F. Baumgartner,
Neerav Karani.

Requirements :

Python 3.6.0
Tensorflow 1.8.0
rest of the requirements are mentioned in the "requirements.txt" file

I)  To clone the git repository.

git clone https://github.com/krishnabits001/task_driven_data_augmentation.git


II) Install python, required packages and tensorflow.

Then, install python packages required using below command or the packages mentioned in the file.
pip install -r requirements.txt

To install tensorflow
pip install tensorflow-gpu=1.8.0

III) Dataset download.

To download the acdc, check the below website.
https://www.creatis.insa-lyon.fr/Challenge/acdc
All the images were bias corrected using N4 algorithm with a threshold value of 0.001. For other details refer to the paper.

IV) Train the models.

One can train the models stepwise (check "train/train_script.sh" script for commands, also stated below)

Steps :
1) To train the deformation field generator cGAN to generate the deformation fields
cd train_model/ 
python tr_deformation_cgan_and_unet.py --dataset=acdc --no_of_tr_imgs=tr1 --comb_tr_imgs=c1 --lr_gen=0.001 --lr_disc=0.001 --ra_en=0 --gan_type=gan --data_aug_seg=1 --ver=0 --en_1hot=1 --lamda_l1_g=0.001 

2) To train the additive intensity field generator cGAN to generate the intensity fields
cd train_model/ 
python tr_intensity_cgan_and_unet.py --dataset=acdc --no_of_tr_imgs=tr1 --comb_tr_imgs=c1 --lr_gen=0.001 --lr_disc=0.001 --ra_en=0 --gan_type=gan --data_aug_seg=1 --ver=0 --en_1hot=1 --lamda_l1_i=0.001 

3) To use both the trained cGANs to generate augmented images and train the unet
cd train_model/ 
python tr_unet_with_deformation_intensity_cgans_augmentations.py --dataset=acdc --no_of_tr_imgs=tr1 --comb_tr_imgs=c1 --lr_gen=0.001 --lr_disc=0.001 --data_aug_seg=1 --ra_en=0 --gan_type=gan --lamda_l1_g=0.001 --lamda_l1_i=0.001 --ver=0 --dsc_loss=0 

To train the baseline with affine transformations for comparison, use the below code file:
cd train_model/ 
python --dataset=acdc --no_of_tr_imgs=tr1 --comb_tr_imgs=c1 --lr_seg=0.001 --ver=0

V) Config files contents.

One can modify the contents of the below 2 config files to run the required experiments.
experiment_init directory contains 2 files.
1) init_acdc.py 
--> contains the config details like target resolution, image dimensions, data path where the dataset is stored and path to save the trained models

2) data_cfg_acdc.py 
--> contains an example of data config details where one can set the patient ids which they want to use as train, val and test images.
