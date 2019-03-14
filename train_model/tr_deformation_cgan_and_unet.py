import os
# # Assign GPU no
#os.environ["CUDA_VISIBLE_DEVICES"]=os.environ['SGE_GPU']
#from tensorflow.python.client import device_lib

import tensorflow as tf
config=tf.ConfigProto()
config.gpu_options.allow_growth=True
config.allow_soft_placement=True

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import time

#to make directories
import pathlib

import sys
sys.path.append('../')

from utils import *

import argparse
parser = argparse.ArgumentParser()
# data set type
parser.add_argument('--dataset', type=str, default='acdc', choices=['acdc'])
# no of training images
parser.add_argument('--no_of_tr_imgs', type=str, default='tr3', choices=['tr1', 'tr3', 'tr5', 'tr15', 'tr40'])
# combination of training images
parser.add_argument('--comb_tr_imgs', type=str, default='c1', choices=['c1', 'c2', 'c3', 'c4', 'c5'])
#learning rate of seg unet
parser.add_argument('--lr_seg', type=float, default=0.001)

# learning rate of generator
parser.add_argument('--lr_gen', type=float, default=0.0001)
# learning rate of discriminator
parser.add_argument('--lr_disc', type=float, default=0.0001)
# lat dim of z sample
parser.add_argument('--z_lat_dim', type=int, default=100)

# ra_en : 0 - disabled, 1 - enabled
parser.add_argument('--ra_en', type=int, default=0)
# select gan type
parser.add_argument('--gan_type', type=str, default='gan', choices=['lsgan', 'gan', 'wgan-gp','ngan'])
# beta value of Adam optimizer
parser.add_argument('--beta_val', type=float, default=0.9)
# to enable the representation of labels with 1 hot encoding
parser.add_argument('--en_1hot', type=float, default=1)

# data aug enable : 0 - disabled, 1 - enabled
parser.add_argument('--data_aug_seg', type=int, default=1, choices=[0,1])

# lamda factors
# for segmenation loss term (lamda_dsc)
parser.add_argument('--lamda_dsc', type=float, default=1)
# adversarial loss term (lamda_adv)
parser.add_argument('--lamda_adv', type=float, default=1)
# for negative L1 loss on spatial transformation (per-pixel flow field/deformation field) term (lamda_l1_g)
parser.add_argument('--lamda_l1_g', type=float, default=0.001)

# version of run
parser.add_argument('--ver', type=int, default=0)

# segmentation loss to optimize
# 0 for weighted cross entropy, 1 for dice score loss
parser.add_argument('--dsc_loss', type=int, default=0)

parse_config = parser.parse_args()

ra_en_val=parse_config.ra_en
if(parse_config.ra_en==1):
    parse_config.ra_en=True
else:
    parse_config.ra_en=False

if parse_config.dataset == 'acdc':
    #print('load acdc configs')
    import experiment_init.init_acdc as cfg
    import experiment_init.data_cfg_acdc as data_list
else:
    raise ValueError(parse_config.dataset)

######################################
# class loaders
# ####################################
#  load dataloader object
from dataloaders import dataloaderObj
dt = dataloaderObj(cfg)

if parse_config.dataset == 'acdc':
    #print('set acdc orig img dataloader handle')
    orig_img_dt=dt.load_acdc_imgs

#  load model object
from models import modelObj
model = modelObj(cfg)
#  load f1_utils object
from f1_utils import f1_utilsObj
f1_util = f1_utilsObj(cfg,dt)

######################################
#define save_dir for the model
save_dir=str(cfg.base_dir)+'/models/'+str(parse_config.dataset)+'/tr_deformation_cgan_unet/ra_en_'+str(ra_en_val)+'_gantype_'+str(parse_config.gan_type)+'/'

if(parse_config.data_aug_seg==0):
    save_dir=str(save_dir)+'no_data_aug/'
    cfg.aug_en=parse_config.data_aug_seg
else:
    save_dir=str(save_dir)+'with_data_aug/'

save_dir=str(save_dir)+'lamda_dsc_'+str(parse_config.lamda_dsc)+'_lamda_adv_'+str(parse_config.lamda_adv)+'_lamda_g_'+str(parse_config.lamda_l1_g)+'/'+\
         str(parse_config.no_of_tr_imgs)+'/'+str(parse_config.comb_tr_imgs)+'_v'+str(parse_config.ver)+\
         '/unet_model_beta1_'+str(parse_config.beta_val)+'_lr_seg_'+str(parse_config.lr_seg)+'_lr_gen_'+str(parse_config.lr_gen)+'_lr_disc_'+str(parse_config.lr_disc)+'/'

print('save_dir',save_dir)
######################################

######################################
# load train and val images
train_list = data_list.train_data(parse_config.no_of_tr_imgs,parse_config.comb_tr_imgs)
#load train data cropped images directly
print('loading train imgs')
train_imgs,train_labels = dt.load_acdc_cropped_img_labels(train_list)

if(parse_config.no_of_tr_imgs=='tr1'):
    train_imgs_copy=np.copy(train_imgs)
    train_labels_copy=np.copy(train_labels)
    while(train_imgs.shape[2]<cfg.batch_size):
        train_imgs=np.concatenate((train_imgs,train_imgs_copy),axis=2)
        train_labels=np.concatenate((train_labels,train_labels_copy),axis=2)
    del train_imgs_copy,train_labels_copy

val_list = data_list.val_data()
#load both val data and its cropped images
print('loading val imgs')
val_label_orig,val_img_crop,val_label_crop,pixel_val_list=load_val_imgs(val_list,dt,orig_img_dt)

# # load unlabeled images
unl_list = data_list.unlabeled_data()
print('loading unlabeled imgs')
unlabeled_imgs=dt.load_acdc_cropped_img_labels(unl_list,label_present=0)
#print('unlabeled_imgs',unlabeled_imgs.shape)

# get test list
print('get test imgs list')
test_list = data_list.test_data()
struct_name=cfg.struct_name
val_step_update=cfg.val_step_update
######################################

######################################

def get_samples(labeled_imgs,unlabeled_imgs):
    # sample z vectors from Gaussian Distribution
    z_samples = np.random.normal(loc=0.0, scale=1.0, size=(cfg.batch_size, parse_config.z_lat_dim)).astype(np.float32)

    # sample Unlabeled data shuffled batch
    unld_img_batch=shuffle_minibatch([unlabeled_imgs],batch_size=int(cfg.batch_size),num_channels=cfg.num_channels,labels_present=0,axis=2)

    # sample Labelled data shuffled batch
    ld_img_batch=shuffle_minibatch([labeled_imgs],batch_size=int(cfg.batch_size),num_channels=cfg.num_channels,labels_present=0,axis=2)

    return z_samples,ld_img_batch,unld_img_batch

def plt_func(sess,ae,save_dir,z_samples,ld_img_batch,unld_img_batch,index=0):
    # plot deformed images for an fixed input image and different per-pixel flow vectors generated from sampled z values
    ld_img_tmp=np.zeros_like(ld_img_batch)
    # select one 2D image from the batch and apply different z's sampled over this selected image
    for i in range(0,20):
        ld_img_tmp[i,:,:,0]=ld_img_batch[index,:,:,0]

    flow_vec,y_geo_deformed,z_cost=sess.run([ae['flow_vec'],ae['y_trans'],ae['z_cost']], feed_dict={ae['x_l']: ld_img_tmp, ae['z']:z_samples,\
                          ae['x_unl']: unld_img_batch, ae['select_mask']: True, ae['train_phase']: False})

    f1_util.plot_deformed_imgs(ld_img_tmp,y_geo_deformed,flow_vec,save_dir,index=index)

    # Plot gif of all the deformed images generated for the fixed input image
    f1_util.write_gif_func(ip_img=y_geo_deformed, imsize=(cfg.img_size_x,cfg.img_size_y),save_dir=save_dir,index=index)


######################################
# Define checkpoint file to save CNN architecture and learnt hyperparameters
checkpoint_filename='unet_'+str(parse_config.dataset)
logs_path = str(save_dir)+'tensorflow_logs/'
best_model_dir=str(save_dir)+'best_model/'
######################################

######################################
# Define deformation field generator model graph
ae = model.spatial_generator_cgan_unet(learn_rate_gen=parse_config.lr_gen,learn_rate_disc=parse_config.lr_disc,\
                        beta1_val=parse_config.beta_val,gan_type=parse_config.gan_type,ra_en=parse_config.ra_en,\
                        learn_rate_seg=parse_config.lr_seg,dsc_loss=parse_config.dsc_loss,en_1hot=parse_config.en_1hot,\
                        lamda_dsc=parse_config.lamda_dsc,lamda_adv=parse_config.lamda_adv,lamda_l1_g=parse_config.lamda_l1_g)

######################################
#  training parameters
start_epoch=0
n_epochs = 10000
disp_step=400
print_step=2000
# no of iterations to train just the segmentation network using the labeled data without any cGAN generated data
seg_tr_limit=400
mean_f1_val_prev=0.1
threshold_f1=0.000001
pathlib.Path(best_model_dir).mkdir(parents=True, exist_ok=True)
######################################

######################################
# define graph to compute deformed image given an per-pixel flow vector and input image
df_ae= model.deform_net()
######################################

######################################
#writer for train summary
train_writer = tf.summary.FileWriter(logs_path)
#writer for dice score and val summary
dsc_writer = tf.summary.FileWriter(logs_path)
val_sum_writer = tf.summary.FileWriter(logs_path)
######################################

######################################
# create a session and initialize variable to use the graph
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())
# Save training data
saver = tf.train.Saver(max_to_keep=2)
######################################

# Run for n_epochs
for epoch_i in range(start_epoch,n_epochs):

    # sample z's from Gaussian Distribution
    z_samples = np.random.normal(loc=0.0, scale=1.0, size=(cfg.batch_size, parse_config.z_lat_dim)).astype(np.float32)

    # sample Unlabeled shuffled batch
    unld_img_batch=shuffle_minibatch([unlabeled_imgs],batch_size=int(cfg.batch_size),num_channels=cfg.num_channels,labels_present=0,axis=2)

    # sample Labelled shuffled batch
    ld_img_batch,ld_label_batch=shuffle_minibatch([train_imgs,train_labels],batch_size=cfg.batch_size,num_channels=cfg.num_channels,axis=2)
    if(cfg.aug_en==1):
        # Apply affine transformations
        ld_img_batch,ld_label_batch=augmentation_function([ld_img_batch,ld_label_batch],dt)
        unld_img_batch=augmentation_function([unld_img_batch],dt,labels_present=0)

    ld_img_batch_tmp=np.copy(ld_img_batch)
    # Compute 1 hot encoding of the segmentation mask labels
    ld_label_batch_1hot = sess.run(df_ae['y_tmp_1hot'],feed_dict={df_ae['y_tmp']:ld_label_batch})

    if(epoch_i>=seg_tr_limit):
        # sample the batch of images and apply deformation field generated by the Generator network on these which are used for the remaining 9500 epochs
        # Batch comprosed of both deformed image,label pairs and original affine transformed image, label pairs
        ld_label_batch_tmp=np.copy(ld_label_batch)
        ###########################
        ## use Deformation field cGAN to generate additional augmented image,label pairs from labeled samples
        flow_vec,ld_img_batch=sess.run([ae['flow_vec'],ae['y_trans']],\
                                    feed_dict={ae['x_l']: ld_img_batch_tmp, ae['z']:z_samples, ae['train_phase']: False})

        ld_label_batch=sess.run([df_ae['deform_y_1hot']],feed_dict={df_ae['y_tmp']:ld_label_batch,df_ae['flow_v']:flow_vec})
        ld_label_batch=ld_label_batch[0]

        ###########################
        #shuffle the quantity/number of images chosen from deformation cGAN augmented images and rest are original images with conventional affine transformations
        no_orig=np.random.randint(5, high=15)
        ld_img_batch[0:no_orig] = ld_img_batch_tmp[0:no_orig]
        if(parse_config.en_1hot==1):
            ld_label_batch[0:no_orig] = ld_label_batch_1hot[0:no_orig]
        else:
            ld_label_batch = np.argmax(ld_label_batch,axis=3)
            ld_label_batch[0:no_orig] = ld_label_batch_tmp[0:no_orig]

        #Pick equal number of images from each category
        # ld_img_batch[0:10]=ld_img_batch_tmp[0:10]
        # ld_label_batch[0:10]=ld_label_batch_1hot[0:10]

    elif(epoch_i<seg_tr_limit):
        # sample only labeled data batches to optimize only Segmentation Network for initial 500 epochs
        ld_img_batch=ld_img_batch
        unld_img_batch=unld_img_batch
        ld_label_batch=ld_label_batch_1hot

    if(epoch_i<seg_tr_limit):
        #Optimize only Segmentation Network for initial 500 epochs
        train_summary,_ =sess.run([ae['seg_summary'],ae['optimizer_unet_seg']], feed_dict={ae['x']: ld_img_batch, ae['y_l']: ld_label_batch,\
                                   ae['select_mask']: False, ae['train_phase']: True})

    if(epoch_i>seg_tr_limit):
        #Optimize Generator (G), Discriminator (D) and Segmentation (S) networks for the remaining 9500 epochs

        # update both Generator and Segmentation Net parameters in the framework using total loss value
        train_summary,_ =sess.run([ae['train_summary'],ae['optimizer_l2_both_gen_unet']], feed_dict={ae['x']: ld_img_batch,ae['x_l']: ld_img_batch,ae['y_l']: ld_label_batch,\
                                   ae['z']:z_samples, ae['x_unl']: unld_img_batch, ae['select_mask']: True, ae['train_phase']: True})
        # update Discriminator Net (D) parameters in the setup using only discriminator loss
        train_summary,_ =sess.run([ae['train_summary'],ae['optimizer_disc']], feed_dict={ae['x']: ld_img_batch,ae['x_l']: ld_img_batch, ae['z']:z_samples,\
                              ae['y_l']: ld_label_batch,ae['x_unl']: unld_img_batch, ae['select_mask']: True, ae['train_phase']: True})

    if(epoch_i%val_step_update==0):
        train_writer.add_summary(train_summary, epoch_i)
        train_writer.flush()

    if(epoch_i%val_step_update==0):
        ##Save the model with best DSC for Validation Image
        mean_f1_arr=[]
        f1_arr=[]
        for val_id_no in range(0,len(val_list)):
            val_img_crop_tmp=val_img_crop[val_id_no]
            val_label_crop_tmp=val_label_crop[val_id_no]
            val_label_orig_tmp=val_label_orig[val_id_no]
            pixel_size_val=pixel_val_list[val_id_no]

            # Compute segmentation mask and dice_score for each validation subject
            pred_sf_mask = f1_util.calc_pred_sf_mask_full(sess, ae, val_img_crop_tmp)
            re_pred_mask_sys,f1_val = f1_util.reshape_img_and_f1_score(pred_sf_mask, val_label_orig_tmp, pixel_size_val)

            #concatenate dice scores of each val image
            mean_f1_arr.append(np.mean(f1_val[1:cfg.num_classes]))
            f1_arr.append(f1_val[1:cfg.num_classes])

        #avg mean over 2 val subjects
        mean_f1_arr=np.asarray(mean_f1_arr)
        mean_f1=np.mean(mean_f1_arr)
        f1_arr=np.asarray(f1_arr)

        if (mean_f1-mean_f1_val_prev>threshold_f1 and epoch_i!=start_epoch):
            print("prev f1_val; present_f1_val", mean_f1_val_prev, mean_f1, mean_f1_arr)
            mean_f1_val_prev = mean_f1
            # to save the best model with maximum dice score over the entire n_epochs
            print("best model saved at epoch no. ", epoch_i)
            mp_best = str(best_model_dir) + str(checkpoint_filename) + '_best_model_epoch_' + str(epoch_i) + '.ckpt'
            saver.save(sess, mp_best)

        #calc. and save validation image dice summary
        dsc_summary_msg = sess.run(ae['val_dsc_summary'], feed_dict={ae['rv_dice']:np.mean(f1_arr[:,0]),\
                                ae['myo_dice']:np.mean(f1_arr[:,1]),ae['lv_dice']:np.mean(f1_arr[:,2]),ae['mean_dice']: mean_f1})
        val_sum_writer.add_summary(dsc_summary_msg, epoch_i)
        val_sum_writer.flush()

    if ((epoch_i==n_epochs-1) and (epoch_i != start_epoch)):
        # model saved at last epoch
        mp = str(save_dir) + str(checkpoint_filename) + '_epochs_' + str(epoch_i) + '.ckpt'
        saver.save(sess, mp)
        try:
            mp_best
        except NameError:
            mp_best=mp

sess.close()
######################################
# restore best model and predict segmentations on test subjects
saver_new = tf.train.Saver()
sess_new = tf.Session(config=config)
saver_new.restore(sess_new, mp_best)
print("best model chkpt name",mp_best)
print("Model restored")

#########################
# To compute inference on test images on the model that yields best dice score on validation images
f1_util.pred_segs_acdc_test_subjs(sess_new,ae,save_dir,orig_img_dt,test_list,struct_name)
#########################
# To plot the generated augmented images from the trained deformation cGAN
for j in range(0,5):
    z_samples,ld_img_batch,unld_img_batch=get_samples(train_imgs,unlabeled_imgs)
    save_dir_tmp=str(save_dir)+'/ep_best_model/'
    plt_func(sess_new,ae,save_dir_tmp,z_samples,ld_img_batch,unld_img_batch,index=j)
######################################
# To compute inference on validation images on the best model
save_dir_tmp=str(save_dir)+'/val_imgs/'
f1_util.pred_segs_acdc_test_subjs(sess_new,ae,save_dir_tmp,orig_img_dt,val_list,struct_name)
######################################
