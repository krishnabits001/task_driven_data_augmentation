**Semi-Supervised and Task-Driven Data Augmentation** <br/>

The code is for the article "Semi-Supervised and Task-Driven Data Augmentation" which got accepted as an oral presentation at IPMI 2019 (26th international conference on Information Processing in Medical Imaging).
The method yields competitive segmentation performance with just 1 labelled training volume.<br/>
https://arxiv.org/abs/1902.05396
https://link.springer.com/chapter/10.1007/978-3-030-20351-1_3


**Authors:** <br/>
Krishna Chaitanya ([email](mailto:krishna.chaitanya@vision.ee.ethz.ch)),<br/>
Christian F. Baumgartner,<br/>
Neerav Karani,<br/>
Ender Konukoglu.<br/>

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
All the images were bias corrected using N4 algorithm with a threshold value of 0.001. For more details, refer to the "bias_correction_details.txt" file.<br/>
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


**Bibtex citation:** <br/>
@InProceedings{10.1007/978-3-030-20351-1_3,
author="Chaitanya, Krishna
and Karani, Neerav
and Baumgartner, Christian F.
and Becker, Anton
and Donati, Olivio
and Konukoglu, Ender",
editor="Chung, Albert C. S.
and Gee, James C.
and Yushkevich, Paul A.
and Bao, Siqi",
title="Semi-supervised and Task-Driven Data Augmentation",
booktitle="Information Processing in Medical Imaging",
year="2019",
publisher="Springer International Publishing",
address="Cham",
pages="29--41",
abstract="Supervised deep learning methods for segmentation require large amounts of labelled training data, without which they are prone to overfitting, not generalizing well to unseen images. In practice, obtaining a large number of annotations from clinical experts is expensive and time-consuming. One way to address scarcity of annotated examples is data augmentation using random spatial and intensity transformations. Recently, it has been proposed to use generative models to synthesize realistic training examples, complementing the random augmentation. So far, these methods have yielded limited gains over the random augmentation. However, there is potential to improve the approach by (i) explicitly modeling deformation fields (non-affine spatial transformation) and intensity transformations and (ii) leveraging unlabelled data during the generative process. With this motivation, we propose a novel task-driven data augmentation method where to synthesize new training examples, a generative network explicitly models and applies deformation fields and additive intensity masks on existing labelled data, modeling shape and intensity variations, respectively. Crucially, the generative model is optimized to be conducive to the task, in this case segmentation, and constrained to match the distribution of images observed from labelled and unlabelled samples. Furthermore, explicit modeling of deformation fields allows synthesizing segmentation masks and images in exact correspondence by simply applying the generated transformation to an input image and the corresponding annotation. Our experiments on cardiac magnetic resonance images (MRI) showed that, for the task of segmentation in small training data scenarios, the proposed method substantially outperforms conventional augmentation techniques.",
isbn="978-3-030-20351-1"
}
<br/>
