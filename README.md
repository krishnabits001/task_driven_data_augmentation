**Semi-Supervised and Task-Driven Data Augmentation** <br/>

The code is for the article "Semi-Supervised and Task-Driven Data Augmentation" which got accepted as an oral presentation at IPMI 2019 (26th international conference on Information Processing in Medical Imaging).
The method yields competitive segmentation performance with just 1 labelled training volume.<br/>
https://arxiv.org/abs/1902.05396

**Authors:** <br/>
Krishna Chaitanya ([email](mailto:krishna.chaitanya@vision.ee.ethz.ch)),<br/>
Christian F. Baumgartner,<br/>
Neerav Karani.<br/>

**Requirements:** <br/>
Python 3.6.0,<br/>
Tensorflow 1.8.0,<br/>
rest of the requirements are mentioned in the "requirements.txt" file. <br/>

I)  To clone the git repository.<br/>
git clone https://github.com/krishnabits001/task_driven_data_augmentation.git <br/>

II) Install python, required packages and tensorflow.<br/>
Then, install python packages required using below command or the packages mentioned in the file.<br/>
pip install -r requirements.txt <br/>

To install tensorflow <br/>
pip install tensorflow-gpu=1.8.0 <br/>

III) Dataset download.<br/>
To download the acdc, check the website :<br/>
https://www.creatis.insa-lyon.fr/Challenge/acdc. <br/>
All the images were bias corrected using N4 algorithm with a threshold value of 0.001. For more details refer to the paper.<br/>
Image and label pairs are re-sampled (to chosen resolution) and cropped/zero-padded to a fixed size using "create_cropped_imgs_acdc.py" file. <br/>

IV) Train the models.<br/>
The models need to be trained sequentially as follows (check "train/train_script.sh" script for commands)<br/>
Steps :<br/>
1) To train the deformation field cGAN to generate the deformation fields.<br/>
cd train_model/ <br/>
python tr_deformation_cgan_and_unet.py --no_of_tr_imgs=tr1 --comb_tr_imgs=c1 --lr_gen=0.001 --lr_disc=0.001 --ra_en=0 --gan_type=gan --data_aug_seg=1 --ver=0 --en_1hot=1 --lamda_l1_g=0.001 <br/>

2) To train the additive intensity field cGAN to generate the intensity fields.<br/>
cd train_model/ <br/>
python tr_intensity_cgan_and_unet.py --no_of_tr_imgs=tr1 --comb_tr_imgs=c1 --lr_gen=0.001 --lr_disc=0.001 --ra_en=0 --gan_type=gan --data_aug_seg=1 --ver=0 --en_1hot=1 --lamda_l1_i=0.001 <br/>

3) To use both the trained cGANs to generate augmented images and train the unet.<br/>
cd train_model/ <br/>
python tr_unet_with_deformation_intensity_cgans_augmentations.py --no_of_tr_imgs=tr1 --comb_tr_imgs=c1 --lr_gen=0.001 --lr_disc=0.001 --data_aug_seg=1 --ra_en=0 --gan_type=gan --lamda_l1_g=0.001 --lamda_l1_i=0.001 --ver=0 --dsc_loss=0 <br/>

To train the baseline with affine transformations for comparison, use the below code file.<br/>
cd train_model/ <br/>
python tr_unet_baseline.py --no_of_tr_imgs=tr1 --comb_tr_imgs=c1 --lr_seg=0.001 --ver=0 <br/>

V) Config files contents.<br/>
One can modify the contents of the below 2 config files to run the required experiments.<br/>
experiment_init directory contains 2 files.<br/>
1) init_acdc.py <br/>
--> contains the config details like target resolution, image dimensions, data path where the dataset is stored and path to save the trained models.<br/>
2) data_cfg_acdc.py <br/>
--> contains an example of data config details where one can set the patient ids which they want to use as train, validation and test images.<br/>
