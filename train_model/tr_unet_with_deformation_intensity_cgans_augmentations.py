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
#data set type
parser.add_argument('--dataset', type=str, default='acdc', choices=['acdc'])
#no of training images
parser.add_argument('--no_of_tr_imgs', type=str, default='tr3', choices=['tr1', 'tr3', 'tr5', 'tr15', 'tr40'])
#combination of training images
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

# lamda factors
# for segmenation loss term (lamda_dsc)
parser.add_argument('--lamda_dsc', type=float, default=1)
# adversarial loss term (lamda_adv)
parser.add_argument('--lamda_adv', type=float, default=1)
### deformation field cGAN specific
# for negative L1 loss on spatial transformation (per-pixel flow field/deformation field) term (lamda_l1_g)
parser.add_argument('--lamda_l1_g', type=float, default=0.001)

### Intensity field cGAN specific
# for negative L1 loss on transformation (additive intensity field) term (lamda_l1_i)
parser.add_argument('--lamda_l1_i', type=float, default=0.001)

#version of run
parser.add_argument('--ver', type=int, default=0)

#data aug - 0 - disabled, 1 - enabled
parser.add_argument('--data_aug_seg', type=int, default=1, choices=[0,1])

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
    #print('set acdc img dataloader handle')
    orig_img_dt=dt.load_acdc_imgs

#  load model object
from models import modelObj
model = modelObj(cfg)

#  load f1_utils object
from f1_utils import f1_utilsObj
f1_util = f1_utilsObj(cfg,dt)

######################################
#define save_dir for the model
proj_save_name='tr_deform_and_int_cgans_data_aug/ra_en_'+str(ra_en_val)+'_gantype_'+str(parse_config.gan_type)+'/'

if(parse_config.data_aug_seg==0):
    save_dir=str(cfg.base_dir)+'/models/'+str(parse_config.dataset)+'/'+str(proj_save_name)+'/no_data_aug/'
    cfg.aug_en=parse_config.data_aug_seg
else:
    save_dir=str(cfg.base_dir)+'/models/'+str(parse_config.dataset)+'/'+str(proj_save_name)+'/with_data_aug/'

save_dir=str(save_dir)+'lamda_dsc_'+str(parse_config.lamda_dsc)+'_lamda_adv_'+str(parse_config.lamda_adv)+\
         '_lamda_g_'+str(parse_config.lamda_l1_g)+'_lamda_i_'+str(parse_config.lamda_l1_i)+'/'+\
         '/'+str(parse_config.no_of_tr_imgs)+'/'+str(parse_config.comb_tr_imgs)+'_v'+str(parse_config.ver)+'/unet_model_dsc_loss_'+str(parse_config.dsc_loss)+'_lr_seg_'+str(parse_config.lr_seg)+'/'
print('save_dir',save_dir)
######################################

######################################
# load train and val images
train_list = data_list.train_data(parse_config.no_of_tr_imgs,parse_config.comb_tr_imgs)
#print(train_list)
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
#print(val_list)
#load both val data and its cropped images
print('loading val imgs')
val_label_orig,val_img_crop,val_label_crop,pixel_val_list=load_val_imgs(val_list,dt,orig_img_dt)
#print(pixel_val_list)

# get test list
print('get test imgs list')
test_list = data_list.test_data()
struct_name=cfg.struct_name
val_step_update=cfg.val_step_update
######################################

######################################
# Define checkpoint file to save CNN architecture and learnt hyperparameters
checkpoint_filename='unet_'+str(parse_config.dataset)
logs_path = str(save_dir)+'tensorflow_logs/'
best_model_dir=str(save_dir)+'best_model/'
######################################

########################################################################
#load deformation field generator net
########################################################################
# Define the model graph
tf.reset_default_graph()
ae_geo = model.spatial_generator_cgan_unet(learn_rate_gen=parse_config.lr_gen,learn_rate_disc=parse_config.lr_disc,\
                        beta1_val=parse_config.beta_val,gan_type=parse_config.gan_type,ra_en=parse_config.ra_en,\
                        learn_rate_seg=parse_config.lr_seg,dsc_loss=parse_config.dsc_loss,en_1hot=parse_config.en_1hot,\
                        lamda_dsc=parse_config.lamda_dsc,lamda_adv=parse_config.lamda_adv,lamda_l1_g=parse_config.lamda_l1_g)

# define model path
model_path=str(cfg.base_dir)+'/models/'+str(parse_config.dataset)+'/tr_deformation_cgan_unet/ra_en_'+str(ra_en_val)+'_gantype_'+str(parse_config.gan_type)+'/'

if(parse_config.data_aug_seg==0):
    model_path=str(model_path)+'no_data_aug/'
    cfg.aug_en=parse_config.data_aug_seg
else:
    model_path=str(model_path)+'with_data_aug/'

model_path=str(model_path)+'lamda_dsc_'+str(parse_config.lamda_dsc)+'_lamda_adv_'+str(parse_config.lamda_adv)+'_lamda_g_'+str(parse_config.lamda_l1_g)+'/'+\
         str(parse_config.no_of_tr_imgs)+'/'+str(parse_config.comb_tr_imgs)+'_v'+str(parse_config.ver)+\
         '/unet_model_beta1_'+str(parse_config.beta_val)+'_lr_seg_'+str(parse_config.lr_seg)+'_lr_gen_'+str(parse_config.lr_gen)+'_lr_disc_'+str(parse_config.lr_disc)+'/'

mp=get_max_chkpt_file(model_path)
print('loading deformation field cGAN checkpoint file',mp)
# create a session and load the parameters learned
saver_geo = tf.train.Saver(max_to_keep=2)
sess_geo = tf.Session(config=config)
saver_geo.restore(sess_geo,mp)
######################################

########################################################################
#load additive intensity field generator net
########################################################################
# Define the model graph
tf.reset_default_graph()
ae_int = model.intensity_transform_cgan_unet(learn_rate_gen=parse_config.lr_gen,learn_rate_disc=parse_config.lr_disc,\
                        beta1_val=parse_config.beta_val,gan_type=parse_config.gan_type,ra_en=parse_config.ra_en,\
                        learn_rate_seg=parse_config.lr_seg,dsc_loss=parse_config.dsc_loss,en_1hot=parse_config.en_1hot,\
                        lamda_dsc=parse_config.lamda_dsc,lamda_adv=parse_config.lamda_adv,lamda_l1_i=parse_config.lamda_l1_i)

# define model path
model_path=str(cfg.base_dir)+'/models/'+str(parse_config.dataset)+'/tr_intensity_cgan_unet/ra_en_'+str(ra_en_val)+'_gantype_'+str(parse_config.gan_type)+'/'

if(parse_config.data_aug_seg==0):
    model_path=str(model_path)+'no_data_aug/'
    cfg.aug_en=parse_config.data_aug_seg
else:
    model_path=str(model_path)+'with_data_aug/'

model_path=str(model_path)+'lamda_dsc_'+str(parse_config.lamda_dsc)+'_lamda_adv_'+str(parse_config.lamda_adv)+'_lamda_i_'+str(parse_config.lamda_l1_i)+'/'+\
         str(parse_config.no_of_tr_imgs)+'/'+str(parse_config.comb_tr_imgs)+'_v'+str(parse_config.ver)+\
         '/unet_model_beta1_'+str(parse_config.beta_val)+'_lr_seg_'+str(parse_config.lr_seg)+'_lr_gen_'+str(parse_config.lr_gen)+'_lr_disc_'+str(parse_config.lr_disc)+'/'

mp=get_max_chkpt_file(model_path)
print('loading additive intensity field cGAN checkpoint file ',mp)
# create a session and load the parameters learned
saver_int = tf.train.Saver(max_to_keep=2)
sess_int = tf.Session(config=config)
saver_int.restore(sess_int,mp)

######################################

######################################
#  training parameters
start_epoch=0
n_epochs = 10000
disp_step=500
mean_f1_val_prev=0.1
threshold_f1=0.00001
debug_en=0
pathlib.Path(best_model_dir).mkdir(parents=True, exist_ok=True)
######################################

######################################
# define current graph - unet
tf.reset_default_graph()
ae = model.unet(learn_rate_seg=parse_config.lr_seg,en_1hot=parse_config.en_1hot,dsc_loss=parse_config.dsc_loss)
######################################

######################################
# define deformations net for labels
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

    #sample Labelled data shuffled batch
    ld_img_batch,ld_label_batch=shuffle_minibatch([train_imgs,train_labels],batch_size=cfg.batch_size,num_channels=cfg.num_channels,axis=2)
    if(cfg.aug_en==1):
        # Apply affine transformations
        ld_img_batch,ld_label_batch=augmentation_function([ld_img_batch,ld_label_batch],dt)

    ld_img_batch_orig_tmp=np.copy(ld_img_batch)
    ld_label_batch_orig_tmp=np.copy(ld_label_batch)
    # Compute 1 hot encoding of the segmentation mask labels
    ld_label_batch_orig_1hot = sess.run(df_ae['y_tmp_1hot'],feed_dict={df_ae['y_tmp']:ld_label_batch_orig_tmp})

    ############################
    ## use Deformation field cGAN to generate additional augmented image,label pairs from labeled samples
    flow_vec,ld_img_batch_geo=sess_geo.run([ae_geo['flow_vec'],ae_geo['y_trans']],\
                                feed_dict={ae_geo['x_l']: ld_img_batch_orig_tmp, ae_geo['z']:z_samples, ae_geo['train_phase']: False})

    ld_label_batch_geo=sess.run([df_ae['deform_y_1hot']],feed_dict={df_ae['y_tmp']:ld_label_batch_orig_tmp,df_ae['flow_v']:flow_vec})
    ld_label_batch_geo=ld_label_batch_geo[0]

    ############################
    # use additive Intensity field cGAN to generate additional augmented image,label pairs from labeled samples
    int_c1,ld_img_batch_int=sess_int.run([ae_int['int_c1'],ae_int['y_int']], feed_dict={ae_int['x']: ld_img_batch_orig_tmp, ae_int['z']:z_samples, ae_int['train_phase']: False})
    ld_label_batch_int = ld_label_batch_orig_1hot

    ############################
    # use additive intensity field cGAN over augmented images generated from deformation field cGAN to create augmented images \
    # that have both spatial deformations and intensity transformations applied in them
    ld_img_batch_geo_tmp=np.copy(ld_img_batch_geo)
    int_c1,ld_img_batch_geo_int=sess_int.run([ae_int['int_c1'],ae_int['y_int']], feed_dict={ae_int['x']: ld_img_batch_geo_tmp, ae_int['z']:z_samples, ae_int['train_phase']: False})
    ld_label_batch_geo_int = np.copy(ld_label_batch_geo)

    # shuffle the quantity/number of images chosen from 
    # deformation field cGAN --> no_g,
    # intensity field cGAN   --> no_i,
    # both cGANs             --> no_b,
    # and rest (batch_size - (no_g+no_i+no_b)) are original images with conventional affine transformations.
    no_g=np.random.randint(1, high=5)
    no_i=np.random.randint(5, high=10)
    no_b=np.random.randint(10, high=15)

    ld_img_batch=ld_img_batch_orig_tmp
    ld_label_batch=ld_label_batch_orig_1hot

    ld_img_batch[0:no_g] = ld_img_batch_geo[0:no_g]
    ld_label_batch[0:no_g] = ld_label_batch_geo[0:no_g]
    ld_img_batch[no_g:no_i] = ld_img_batch_int[no_g:no_i]
    ld_label_batch[no_g:no_i] = ld_label_batch_int[no_g:no_i]
    ld_img_batch[no_i:no_b] = ld_img_batch_geo_int[no_i:no_b]
    ld_label_batch[no_i:no_b] = ld_label_batch_geo_int[no_i:no_b]

    #Optimer over this batch of images
    train_summary,_ =sess.run([ae['train_summary'],ae['optimizer_unet_seg']], feed_dict={ae['x']: ld_img_batch, ae['y_l']: ld_label_batch,\
                               ae['select_mask']: False, ae['train_phase']: True})

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
print("best model chkpt",mp_best)
print("Model restored")

#########################
# To compute inference on test images on the model that yields best dice score on validation images
f1_util.pred_segs_acdc_test_subjs(sess_new,ae,save_dir,orig_img_dt,test_list,struct_name)
######################################
# To compute inference on validation images on the best model
save_dir_tmp=str(save_dir)+'/val_imgs/'
f1_util.pred_segs_acdc_test_subjs(sess_new,ae,save_dir_tmp,orig_img_dt,val_list,struct_name)
######################################
