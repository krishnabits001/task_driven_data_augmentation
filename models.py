#contains models for segmentation project - SSL

import tensorflow as tf
import numpy as np

# Load layers and losses
from layers_bn import layersObj
layers = layersObj()

from losses import lossObj
loss = lossObj()

class modelObj:
    def __init__(self,cfg):
        self.img_size_x=cfg.img_size_x
        self.img_size_y=cfg.img_size_y
        self.num_classes=cfg.num_classes
        self.method_val = cfg.method_val
        self.img_size_flat=cfg.img_size_flat
        self.batch_size=cfg.batch_size

    def deform_net(self):
        # placeholders for the network
        x_tmp = tf.placeholder(tf.float32, shape=[self.batch_size, self.img_size_x, self.img_size_y, 1], name='x_tmp')
        v_tmp = tf.placeholder(tf.float32, shape=[self.batch_size, self.img_size_x, self.img_size_y, 2], name='v_tmp')
        y_tmp = tf.placeholder(tf.int32, shape=[self.batch_size, self.img_size_x, self.img_size_y], name='y_tmp')

        y_tmp_1hot = tf.one_hot(y_tmp,depth=self.num_classes)
        w_tmp = tf.contrib.image.dense_image_warp(image=x_tmp,flow=v_tmp,name='dense_image_warp_tmp')
        w_tmp_1hot = tf.contrib.image.dense_image_warp(image=y_tmp_1hot,flow=v_tmp,name='dense_image_warp_tmp_1hot')

        return {'x_tmp':x_tmp,'flow_v':v_tmp,'deform_x':w_tmp,'y_tmp':y_tmp,'y_tmp_1hot':y_tmp_1hot,'deform_y_1hot':w_tmp_1hot}

    def contrast_net(self):
        # placeholders for the network
        x_tmp = tf.placeholder(tf.float32, shape=[self.batch_size, self.img_size_x, self.img_size_y, 1], name='x_tmp')

        rd_cont = tf.image.random_contrast(x_tmp,lower=0.8,upper=1.2,seed=1)
        rd_brit = tf.image.random_brightness(x_tmp,max_delta=0.1,seed=1)
        c_ind=np.arange(0,int(self.batch_size/2),dtype=np.int32)
        b_ind=np.arange(int(self.batch_size/2),int(self.batch_size),dtype=np.int32)

        rd_fin = tf.concat((tf.gather(rd_cont,c_ind),tf.gather(rd_brit,b_ind)),axis=0)

        return {'x_tmp':x_tmp,'rd_fin':rd_fin,'rd_cont':rd_cont,'rd_brit':rd_brit}


    def unet(self,learn_rate_seg=0.001,fs_de=2,dsc_loss=1,en_1hot=0):

        no_filters=[1, 16, 32, 64, 128, 256]
        #default U-Net filters
        #no_filters = [1, 64, 128, 256, 512, 1024]

        if(self.num_classes==2):
            class_weights = tf.constant([[0.05, 0.95]],name='class_weights')
        elif(self.num_classes==3):
            class_weights = tf.constant([[0.05, 0.5, 0.45]],name='class_weights')
        elif(self.num_classes==4):
            class_weights = tf.constant([[0.1, 0.3, 0.3, 0.3]],name='class_weights')

        num_channels=no_filters[0]
        # placeholders for the network
        # Inputs
        x = tf.placeholder(tf.float32, shape=[None, self.img_size_x, self.img_size_y, num_channels], name='x')
        if(en_1hot==1):
            y_l = tf.placeholder(tf.float32, shape=[None, self.img_size_x, self.img_size_y,self.num_classes], name='y_l')
        else:
            y_l = tf.placeholder(tf.int32, shape=[None, self.img_size_x, self.img_size_y], name='y_l')
        select_mask = tf.placeholder(tf.bool, name='select_mask')
        train_phase = tf.placeholder(tf.bool, name='train_phase')

        if(en_1hot==0):
            y_l_onehot=tf.one_hot(y_l,depth=self.num_classes)
        else:
            y_l_onehot=y_l

    ############################################
        #U-Net like Network
    ############################################
        #Encoder - Downsampling Path
    ############################################
        # 2x 3x3 conv and 1 maxpool
        # Level 1
        enc_c1_a = layers.conv2d_layer(ip_layer=x,name='enc_c1_a', num_filters=no_filters[1], use_relu=True, use_batch_norm=True, training_phase=train_phase)
        enc_c1_b = layers.conv2d_layer(ip_layer=enc_c1_a, name='enc_c1_b', num_filters=no_filters[1], use_relu=True, use_batch_norm=True, training_phase=train_phase)
        enc_c1_pool = layers.max_pool_layer2d(enc_c1_b, kernel_size=(2, 2), strides=(2, 2), padding="SAME", name='enc_c1_pool')

        # Level 2
        enc_c2_a = layers.conv2d_layer(ip_layer=enc_c1_pool,name='enc_c2_a', num_filters=no_filters[2], use_relu=True, use_batch_norm=True, training_phase=train_phase)
        enc_c2_b = layers.conv2d_layer(ip_layer=enc_c2_a, name='enc_c2_b', num_filters=no_filters[2], use_relu=True, use_batch_norm=True, training_phase=train_phase)
        enc_c2_pool = layers.max_pool_layer2d(enc_c2_b, kernel_size=(2, 2), strides=(2, 2), padding="SAME", name='enc_c2_pool')

        # Level 3
        enc_c3_a = layers.conv2d_layer(ip_layer=enc_c2_pool,name='enc_c3_a', num_filters=no_filters[3], use_relu=True, use_batch_norm=True, training_phase=train_phase)
        enc_c3_b = layers.conv2d_layer(ip_layer=enc_c3_a, name='enc_c3_b', num_filters=no_filters[3], use_relu=True, use_batch_norm=True, training_phase=train_phase)
        enc_c3_pool = layers.max_pool_layer2d(enc_c3_b, kernel_size=(2, 2), strides=(2, 2), padding="SAME", name='enc_c3_pool')

        # Level 4
        enc_c4_a = layers.conv2d_layer(ip_layer=enc_c3_pool,name='enc_c4_a', num_filters=no_filters[4], use_relu=True, use_batch_norm=True, training_phase=train_phase)
        enc_c4_b = layers.conv2d_layer(ip_layer=enc_c4_a, name='enc_c4_b', num_filters=no_filters[4], use_relu=True, use_batch_norm=True, training_phase=train_phase)
        enc_c4_pool = layers.max_pool_layer2d(enc_c4_b, kernel_size=(2, 2), strides=(2, 2), padding="SAME", name='enc_c4_pool')

        # Level 5 - 2x Conv
        enc_c5_a = layers.conv2d_layer(ip_layer=enc_c4_pool,name='enc_c5_a', num_filters=no_filters[5], use_relu=True, use_batch_norm=True, training_phase=train_phase)
        enc_c5_b = layers.conv2d_layer(ip_layer=enc_c5_a, name='enc_c5_b', num_filters=no_filters[5], use_relu=True, use_batch_norm=True, training_phase=train_phase)

        ########################
        # Decoder - Upsampling Path
        ########################
        #Upsample + 2x2 conv to half the no. of feature channels + SKIP connection (concate the conv. layers)
        # Level 5 - 1 upsampling layer + 1 conv op. + skip connection + 2x conv op.
        scale_val=2
        dec_up5 = layers.upsample_layer(ip_layer=enc_c5_b, method=self.method_val, scale_factor=scale_val)
        dec_dc5 = layers.conv2d_layer(ip_layer=dec_up5,name='dec_dc5', kernel_size=(fs_de,fs_de),num_filters=no_filters[4], use_relu=True, use_batch_norm=True, training_phase=train_phase)
        dec_cat_c5 = tf.concat((dec_dc5,enc_c4_b),axis=3,name='dec_cat_c5')
        dec_c4_a = layers.conv2d_layer(ip_layer=dec_cat_c5,name='dec_c4_a', num_filters=no_filters[4], use_relu=True, use_batch_norm=True, training_phase=train_phase)
        dec_c4_b = layers.conv2d_layer(ip_layer=dec_c4_a,name='dec_c4_b', num_filters=no_filters[4], use_relu=True, use_batch_norm=True, training_phase=train_phase)

        # Level 4
        dec_up4 = layers.upsample_layer(ip_layer=dec_c4_b, method=self.method_val, scale_factor=scale_val)
        dec_dc4 = layers.conv2d_layer(ip_layer=dec_up4,name='dec_dc4', kernel_size=(fs_de,fs_de),num_filters=no_filters[3], use_relu=True, use_batch_norm=True, training_phase=train_phase)
        dec_cat_c4 = tf.concat((dec_dc4,enc_c3_b),axis=3,name='dec_cat_c4')
        dec_c3_a = layers.conv2d_layer(ip_layer=dec_cat_c4,name='dec_c3_a', num_filters=no_filters[3], use_relu=True, use_batch_norm=True, training_phase=train_phase)
        dec_c3_b = layers.conv2d_layer(ip_layer=dec_c3_a,name='dec_c3_b', num_filters=no_filters[3], use_relu=True, use_batch_norm=True, training_phase=train_phase)

        # Level 3
        dec_up3 = layers.upsample_layer(ip_layer=dec_c3_b, method=self.method_val, scale_factor=scale_val)
        dec_dc3 = layers.conv2d_layer(ip_layer=dec_up3,name='dec_dc3', kernel_size=(fs_de,fs_de),num_filters=no_filters[2],use_relu=True, use_batch_norm=True, training_phase=train_phase)
        dec_cat_c3 = tf.concat((dec_dc3,enc_c2_b),axis=3,name='dec_cat_c3')
        dec_c2_a = layers.conv2d_layer(ip_layer=dec_cat_c3,name='dec_c2_a', num_filters=no_filters[2], use_relu=True, use_batch_norm=True, training_phase=train_phase)
        dec_c2_b = layers.conv2d_layer(ip_layer=dec_c2_a,name='dec_c2_b', num_filters=no_filters[2], use_relu=True, use_batch_norm=True, training_phase=train_phase)

        # Level 2
        dec_up2 = layers.upsample_layer(ip_layer=dec_c2_b, method=self.method_val, scale_factor=scale_val)
        dec_dc2 = layers.conv2d_layer(ip_layer=dec_up2,name='dec_dc2', kernel_size=(fs_de,fs_de),num_filters=no_filters[1], use_relu=True, use_batch_norm=True, training_phase=train_phase)
        dec_cat_c2 = tf.concat((dec_dc2,enc_c1_b),axis=3,name='dec_cat_c2')
        dec_c1_a = layers.conv2d_layer(ip_layer=dec_cat_c2,name='dec_c1_a', num_filters=no_filters[1], use_relu=True, use_batch_norm=True, training_phase=train_phase)

        # Level 1
        seg_c1_a = layers.conv2d_layer(ip_layer=dec_c1_a,name='seg_c1_a',num_filters=no_filters[1], use_relu=True, use_batch_norm=True, training_phase=train_phase)
        seg_c1_b = layers.conv2d_layer(ip_layer=seg_c1_a,name='seg_c1_b', num_filters=no_filters[1], use_relu=True, use_batch_norm=True, training_phase=train_phase)
        seg_c1_c = layers.conv2d_layer(ip_layer=seg_c1_b,name='seg_c1_c', num_filters=no_filters[1], use_relu=True, use_batch_norm=True, training_phase=train_phase)

        #Final output layer - Logits before softmax
        seg_fin_layer = layers.conv2d_layer(ip_layer=seg_c1_c,name='seg_fin_layer', num_filters=self.num_classes,use_relu=False, use_batch_norm=False, training_phase=train_phase)

        # Predict Class
        y_pred = tf.nn.softmax(seg_fin_layer)
        y_pred_cls = tf.argmax(y_pred,axis=3)

        ########################
        # Simple Cross Entropy (CE) between predicted labels and true labels
        if(dsc_loss==1):
            # For dice score loss function
            #without background
            seg_cost = loss.dice_loss_without_backgrnd(logits=seg_fin_layer, labels=y_l_onehot)
            #with background
            #seg_cost = dice_loss_with_backgrnd(logits=seg_fin_layer, labels=y_l_onehot)
        else:
            # For Weighted Cross Entropy loss function with background
            seg_cost = loss.pixel_wise_cross_entropy_loss_weighted_nn(logits=seg_fin_layer, labels=y_l_onehot, class_weights=class_weights)

        # var list of u-net (segmentation net)
        seg_net_vars = []
        for v in tf.trainable_variables():
            var_name = v.name
            if 'enc_' in var_name: seg_net_vars.append(v)
            elif 'dec_' in var_name: seg_net_vars.append(v)
            elif 'seg_' in var_name: seg_net_vars.append(v)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            cost_seg=tf.reduce_mean(seg_cost)
            optimizer_unet_seg = tf.train.AdamOptimizer(learn_rate_seg).minimize(cost_seg,var_list=seg_net_vars)

        seg_summary = tf.summary.scalar('seg_cost', tf.reduce_mean(seg_cost))
        # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
        train_summary = tf.summary.merge([seg_summary])
        # For dice score summary
        rv_dice = tf.placeholder(tf.float32, shape=[], name='rv_dice')
        rv_dice_summary = tf.summary.scalar('rv_val_dice', rv_dice)
        myo_dice = tf.placeholder(tf.float32, shape=[], name='myo_dice')
        myo_dice_summary = tf.summary.scalar('myo_val_dice', myo_dice)
        lv_dice = tf.placeholder(tf.float32, shape=[], name='lv_dice')
        lv_dice_summary = tf.summary.scalar('lv_val_dice', lv_dice)

        mean_dice = tf.placeholder(tf.float32, shape=[], name='mean_dice')
        mean_dice_summary = tf.summary.scalar('mean_val_dice', mean_dice)

        val_dsc_summary = tf.summary.merge([mean_dice_summary,rv_dice_summary,myo_dice_summary,lv_dice_summary])

        val_totalc = tf.placeholder(tf.float32, shape=[], name='val_totalc')
        val_totalc_sum= tf.summary.scalar('val_totalc_', val_totalc)
        val_summary = tf.summary.merge([val_totalc_sum])

        return {'x': x, 'y_l':y_l, 'train_phase':train_phase,'select_mask': select_mask,'seg_cost': cost_seg, \
                'y_pred' : y_pred, 'y_pred_cls': y_pred_cls, 'optimizer_unet_seg':optimizer_unet_seg,\
                'train_summary':train_summary,'seg_fin_layer':seg_fin_layer, \
                'rv_dice':rv_dice,'myo_dice':myo_dice,'lv_dice':lv_dice,'mean_dice':mean_dice,'val_dsc_summary':val_dsc_summary,\
                'val_totalc':val_totalc,'val_summary':val_summary}

    def discriminator_loss(self, Ra, loss_func, real, fake):
        real_loss = 0
        fake_loss = 0

        if Ra and loss_func.__contains__('wgan') :
            #print("No exist [Ra + WGAN], so use the {} loss function".format(loss_func))
            Ra = False

        if Ra :
            real_logit = (real - tf.reduce_mean(fake))
            fake_logit = (fake - tf.reduce_mean(real))

            if loss_func == 'lsgan' :
                real_loss = tf.reduce_mean(tf.square(real_logit - 1.0))
                fake_loss = tf.reduce_mean(tf.square(fake_logit + 1.0))

            if loss_func == 'gan' or loss_func == 'gan-gp' or loss_func == 'dragan' :
                print('D ra_en sigmoid loss')
                real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(real), logits=real_logit))
                fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(fake), logits=fake_logit))

            if loss_func == 'ngan' :
                print('D ra_en softmax loss')
                real_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.ones_like(real), logits=real_logit))
                fake_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.zeros_like(fake), logits=fake_logit))

            if loss_func == 'hinge' :
                real_loss = tf.reduce_mean(tf.nn.relu(1.0 - real_logit))
                fake_loss = tf.reduce_mean(tf.nn.relu(1.0 + fake_logit))

        else :
            if loss_func == 'wgan-gp' or loss_func == 'wgan-lp' :
                real_loss = -tf.reduce_mean(real)
                fake_loss = tf.reduce_mean(fake)

            if loss_func == 'lsgan' :
                real_loss = tf.reduce_mean(tf.square(real - 1.0))
                fake_loss = tf.reduce_mean(tf.square(fake))

            if loss_func == 'gan' or loss_func == 'gan-gp' or loss_func == 'dragan' :
                print('D ra_off sigmoid loss')
                real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(real), logits=real))
                fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(fake), logits=fake))

            if loss_func == 'ngan':
                print('D ra_off softmax loss')
                real_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.ones_like(real), logits=real))
                fake_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.zeros_like(fake), logits=fake))

            if loss_func == 'hinge' :
                real_loss = tf.reduce_mean(tf.nn.relu(1.0 - real))
                fake_loss = tf.reduce_mean(tf.nn.relu(1.0 + fake))

        loss = real_loss + fake_loss

        return loss

    def generator_loss(self, Ra, loss_func, real, fake):
        fake_loss = 0
        real_loss = 0

        if Ra and loss_func.__contains__('wgan') :
            #print("No exist [Ra + WGAN], so use the {} loss function".format(loss_func))
            Ra = False

        if Ra :
            fake_logit = (fake - tf.reduce_mean(real))
            real_logit = (real - tf.reduce_mean(fake))

            if loss_func == 'lsgan' :
                fake_loss = tf.reduce_mean(tf.square(fake_logit - 1.0))
                real_loss = tf.reduce_mean(tf.square(real_logit + 1.0))

            if loss_func == 'gan' or loss_func == 'gan-gp' or loss_func == 'dragan' :
                print('G ra_en sigmoid loss')
                fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(fake), logits=fake_logit))
                real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(real), logits=real_logit))

            if loss_func == 'ngan' :
                print('G ra_en softmax loss')
                fake_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.ones_like(fake), logits=fake_logit))
                real_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.zeros_like(real), logits=real_logit))

            if loss_func == 'hinge' :
                fake_loss = tf.reduce_mean(tf.nn.relu(1.0 - fake_logit))
                real_loss = tf.reduce_mean(tf.nn.relu(1.0 + real_logit))

        else :
            if loss_func == 'wgan-gp' or loss_func == 'wgan-lp':
                fake_loss = -tf.reduce_mean(fake)

            if loss_func == 'lsgan' :
                fake_loss = tf.reduce_mean(tf.square(fake - 1.0))

            if loss_func == 'gan' or loss_func == 'gan-gp' or loss_func == 'dragan' :
                print('G ra_off sigmoid loss')
                fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(fake), logits=fake))

            if loss_func == 'ngan':
                print('G ra_off softmax loss')
                fake_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.ones_like(fake), logits=fake))

            if loss_func == 'hinge' :
                fake_loss = -tf.reduce_mean(fake)

        loss = fake_loss + real_loss

        return loss

    def spatial_generator_cgan_unet(self,learn_rate_gen=0.0001,learn_rate_disc=0.0001,z_lat_dim=100,lat_dim=128,beta1_val=0.9,\
                     gan_type='gan',ra_en=True,learn_rate_seg=0.001,dsc_loss=1,en_1hot=0,lamda_dsc=1,lamda_adv=1,lamda_l1_g=1):


        no_filters=[1, 16, 32, 64, 128, 256]

        if(self.num_classes==2):
            class_weights = tf.constant([[0.05, 0.95]],name='class_weights')
        elif(self.num_classes==3):
            class_weights = tf.constant([[0.05, 0.5, 0.45]],name='class_weights')
        elif(self.num_classes==4):
            class_weights = tf.constant([[0.1, 0.3, 0.3, 0.3]],name='class_weights')

        std_init=0.01
        SEED=1
        acti='xavier'
        intl=tf.truncated_normal_initializer(stddev=std_init, seed=SEED)

        hid_dim=int(self.img_size_x*self.img_size_y*no_filters[4]/(32*32))
        latent_dim=lat_dim

        z_hid_dim=int(self.img_size_x*self.img_size_y*no_filters[4]/(32*32))
        dim_x = int(self.img_size_x/32)

        #Generator - FCN variables
        gen_c1_weights = tf.get_variable(name="gen_c1_weights",shape=[z_lat_dim,z_hid_dim], initializer=intl)
        gen_c1_biases = tf.get_variable(name="gen_c1_biases",shape=[z_hid_dim], initializer=tf.constant_initializer(value=0))

        #Discriminator - FCN variables
        fcn_c1_weights = tf.get_variable(name="fcn_c1_weights",shape=[hid_dim, latent_dim], initializer=intl)
        fcn_c1_biases = tf.get_variable(name="fcn_c1_biases",shape=[latent_dim], initializer=tf.constant_initializer(value=0))
        fcn_c2_weights = tf.get_variable(name="fcn_c2_weights",shape=[latent_dim, latent_dim], initializer=intl)
        fcn_c2_biases = tf.get_variable(name="fcn_c2_biases",shape=[latent_dim], initializer=tf.constant_initializer(value=0))
        fcn_c3_weights = tf.get_variable(name="fcn_c3_weights",shape=[latent_dim,1], initializer=intl)
        fcn_c3_biases = tf.get_variable(name="fcn_c3_biases",shape=[1], initializer=tf.constant_initializer(value=0))


        num_channels=no_filters[0]
        # Placeholders
        # input to the network
        z = tf.placeholder(tf.float32, shape=[self.batch_size, z_lat_dim], name='z')
        x_l = tf.placeholder(tf.float32, shape=[self.batch_size, self.img_size_x, self.img_size_y, num_channels], name='x_l')
        x = tf.placeholder(tf.float32, shape=[None, self.img_size_x, self.img_size_y, num_channels], name='x')
        x_unl = tf.placeholder(tf.float32, shape=[None, self.img_size_x, self.img_size_y, num_channels], name='x_unl')
        if(en_1hot==1):
            y_l = tf.placeholder(tf.float32, shape=[None, self.img_size_x, self.img_size_y,self.num_classes], name='y_l')
        else:
            y_l = tf.placeholder(tf.int32, shape=[None, self.img_size_x, self.img_size_y], name='y_l')
        select_mask = tf.placeholder(tf.bool, name='select_mask')
        train_phase = tf.placeholder(tf.bool, name='train_phase')
        if(en_1hot==0):
            y_l_onehot=tf.one_hot(y_l,depth=self.num_classes)
        else:
            y_l_onehot=y_l

        ############################################
        ## Generator Network
        ############################################
        # Dense layer + Reshape reshape to down sampled image dimensions
        gen_fcn_c1 = tf.matmul(z, gen_c1_weights) + gen_c1_biases
        gen_fcn_relu_c1 = tf.nn.relu(gen_fcn_c1)
        gen_fcn_reshaped = tf.reshape(gen_fcn_relu_c1 ,[-1,dim_x,dim_x,no_filters[4]])

        # Level 5 - Upsampling layer + Conv. layer
        fs_de=2
        scale_val=2
        gen_up5 = layers.upsample_layer(ip_layer=gen_fcn_reshaped, method=self.method_val, scale_factor=scale_val)
        gen_c5 = layers.conv2d_layer(ip_layer=gen_up5,name='gen_c5', kernel_size=(fs_de,fs_de),num_filters=no_filters[4],use_relu=True, use_batch_norm=True, training_phase=train_phase)

        # Level 4
        gen_up4 = layers.upsample_layer(ip_layer=gen_c5, method=self.method_val, scale_factor=scale_val)
        gen_c4 = layers.conv2d_layer(ip_layer=gen_up4,name='gen_c4', kernel_size=(fs_de,fs_de),num_filters=no_filters[3],use_relu=True, use_batch_norm=True, training_phase=train_phase)

        # Level 3
        gen_up3 = layers.upsample_layer(ip_layer=gen_c4, method=self.method_val, scale_factor=scale_val)
        gen_c3 = layers.conv2d_layer(ip_layer=gen_up3,name='gen_c3', kernel_size=(fs_de,fs_de),num_filters=no_filters[2],use_relu=True, use_batch_norm=True, training_phase=train_phase)

        # Level 2
        gen_up2 = layers.upsample_layer(ip_layer=gen_c3, method=self.method_val, scale_factor=scale_val)
        gen_c2 = layers.conv2d_layer(ip_layer=gen_up2,name='gen_c2', kernel_size=(fs_de,fs_de),num_filters=no_filters[1],use_relu=True, use_batch_norm=True, training_phase=train_phase)

        # Level 1
        gen_up1 = layers.upsample_layer(ip_layer=gen_c2, method=self.method_val, scale_factor=scale_val)
        gen_c1 = layers.conv2d_layer(ip_layer=gen_up1,name='gen_c1', kernel_size=(fs_de,fs_de),num_filters=no_filters[1],use_relu=False, use_batch_norm=False, training_phase=train_phase)

        # Conv. ops on input image
        conv_1a = layers.conv2d_layer(ip_layer=x_l,name='conv_1a',num_filters=no_filters[1], use_relu=True, use_batch_norm=True, training_phase=train_phase)
        conv_1b = layers.conv2d_layer(ip_layer=conv_1a,name='conv_1b',num_filters=no_filters[1], use_relu=True, use_batch_norm=True, training_phase=train_phase)

        # Concatenate features obtained by conv. ops on image and on 'z'
        gen_cat=tf.concat((gen_c1,conv_1b),axis=3)

        # More Conv. ops on concatenated feature maps
        conv_1c = layers.conv2d_layer(ip_layer=gen_cat,name='conv_1c',num_filters=no_filters[1], use_relu=True, use_batch_norm=True, training_phase=train_phase)
        conv_1d = layers.conv2d_layer(ip_layer=conv_1c,name='conv_1d',num_filters=no_filters[1], use_relu=True, use_batch_norm=True, training_phase=train_phase)
        conv_1e = layers.conv2d_layer(ip_layer=conv_1d,name='conv_1e',num_filters=2, use_relu=False, use_batch_norm=False, training_phase=train_phase)

        flow_vec = conv_1e

        # apply flow vector on the input image to get non-affine transformed image
        y_trans=tf.contrib.image.dense_image_warp(image=x_l,flow=flow_vec,name='dense_image_warp')


        ############################################
        ## Discriminator Network
        ############################################

        cat_disc_c1=tf.concat((y_trans,x_unl),axis=0,name='cat_disc_c1')

        # Choose between concate or true+gen images or gen images
        cat_disc_c1 = tf.cond(select_mask,lambda : cat_disc_c1, lambda :y_trans)

        # DISC Net Architecutre
        disc_c1 = layers.conv2d_layer(ip_layer=cat_disc_c1, name='disc_c1', num_filters=no_filters[1],kernel_size=(5, 5),strides=(2, 2),use_relu=False, use_batch_norm=True, training_phase=train_phase)
        disc_c1 = tf.nn.leaky_relu(disc_c1, alpha=0.2)

        disc_c2 = layers.conv2d_layer(ip_layer=disc_c1, name='disc_c2', num_filters=no_filters[2],kernel_size=(5, 5),strides=(2, 2),use_relu=False, use_batch_norm=True, training_phase=train_phase)
        disc_c2 = tf.nn.leaky_relu(disc_c2, alpha=0.2)

        disc_c3 = layers.conv2d_layer(ip_layer=disc_c2, name='disc_c3', num_filters=no_filters[3],kernel_size=(5, 5),strides=(2, 2),use_relu=False, use_batch_norm=True, training_phase=train_phase)
        disc_c3 = tf.nn.leaky_relu(disc_c3, alpha=0.2)

        disc_c4 = layers.conv2d_layer(ip_layer=disc_c3, name='disc_c4', num_filters=no_filters[4],kernel_size=(5, 5),strides=(2, 2),use_relu=False, use_batch_norm=True, training_phase=train_phase)
        disc_c4 = tf.nn.leaky_relu(disc_c4, alpha=0.2)

        disc_c5 = layers.conv2d_layer(ip_layer=disc_c4, name='disc_c5', num_filters=no_filters[4],kernel_size=(5, 5),strides=(2, 2),use_relu=False, use_batch_norm=True, training_phase=train_phase)
        disc_c5_pool = tf.nn.leaky_relu(disc_c5, alpha=0.2)

        # Flat conv for FCN
        flat_conv = tf.contrib.layers.flatten(disc_c5_pool)

        # FCN + Relu - x2
        z_fcn_c1 = tf.matmul(flat_conv, fcn_c1_weights) + fcn_c1_biases
        z_fcn_relu_c1 = tf.nn.leaky_relu(z_fcn_c1)

        z_fcn_c2 = tf.matmul(z_fcn_relu_c1, fcn_c2_weights) + fcn_c2_biases
        z_fcn_relu_c2 = tf.nn.leaky_relu(z_fcn_c2)

        # 1 fully connected layer to determine input images into real / fake categories
        z_class = tf.matmul(z_fcn_relu_c2, fcn_c3_weights) + fcn_c3_biases

        z_pred=z_class
        z_pred_cls=z_pred

        fake_indices=np.arange(0,self.batch_size,dtype=np.int32)
        fake = tf.gather(z_class,fake_indices)
        real_indices=np.arange(self.batch_size,2*self.batch_size,dtype=np.int32)
        real = tf.gather(z_class,real_indices)

        # Discriminator loss
        z_cost=self.discriminator_loss(Ra=ra_en, loss_func=gan_type, real=real, fake=fake)
        # Generator loss
        g_cost=self.generator_loss(Ra=ra_en, loss_func=gan_type, real=real, fake=fake)

        # divide the var list into Generator Network and Discriminator Net
        gen_net_vars = []
        disc_net_vars = []
        for v in tf.trainable_variables():
            var_name = v.name
            if 'disc_' in var_name: disc_net_vars.append(v)
            elif 'fcn_' in var_name: disc_net_vars.append(v)
            elif 'gen_' in var_name: gen_net_vars.append(v)
            elif 'conv_' in var_name: gen_net_vars.append(v)


    ############################################
        #U-Net Network
    ############################################
        # Encoder - Downsampling Path
        ############################################
        # 2x 3x3 conv and 1 maxpool
        # Level 1
        enc_c1_a = layers.conv2d_layer(ip_layer=x,name='enc_c1_a', num_filters=no_filters[1], use_relu=True, use_batch_norm=True, training_phase=train_phase)
        enc_c1_b = layers.conv2d_layer(ip_layer=enc_c1_a, name='enc_c1_b', num_filters=no_filters[1], use_relu=True, use_batch_norm=True, training_phase=train_phase)
        enc_c1_pool = layers.max_pool_layer2d(enc_c1_b, kernel_size=(2, 2), strides=(2, 2), padding="SAME", name='enc_c1_pool')

        # Level 2
        enc_c2_a = layers.conv2d_layer(ip_layer=enc_c1_pool,name='enc_c2_a', num_filters=no_filters[2], use_relu=True, use_batch_norm=True, training_phase=train_phase)
        enc_c2_b = layers.conv2d_layer(ip_layer=enc_c2_a, name='enc_c2_b', num_filters=no_filters[2], use_relu=True, use_batch_norm=True, training_phase=train_phase)
        enc_c2_pool = layers.max_pool_layer2d(enc_c2_b, kernel_size=(2, 2), strides=(2, 2), padding="SAME", name='enc_c2_pool')

        # Level 3
        enc_c3_a = layers.conv2d_layer(ip_layer=enc_c2_pool,name='enc_c3_a', num_filters=no_filters[3], use_relu=True, use_batch_norm=True, training_phase=train_phase)
        enc_c3_b = layers.conv2d_layer(ip_layer=enc_c3_a, name='enc_c3_b', num_filters=no_filters[3], use_relu=True, use_batch_norm=True, training_phase=train_phase)
        enc_c3_pool = layers.max_pool_layer2d(enc_c3_b, kernel_size=(2, 2), strides=(2, 2), padding="SAME", name='enc_c3_pool')

        # Level 4
        enc_c4_a = layers.conv2d_layer(ip_layer=enc_c3_pool,name='enc_c4_a', num_filters=no_filters[4], use_relu=True, use_batch_norm=True, training_phase=train_phase)
        enc_c4_b = layers.conv2d_layer(ip_layer=enc_c4_a, name='enc_c4_b', num_filters=no_filters[4], use_relu=True, use_batch_norm=True, training_phase=train_phase)
        enc_c4_pool = layers.max_pool_layer2d(enc_c4_b, kernel_size=(2, 2), strides=(2, 2), padding="SAME", name='enc_c4_pool')

        # Level 5 - 2x Conv
        enc_c5_a = layers.conv2d_layer(ip_layer=enc_c4_pool,name='enc_c5_a', num_filters=no_filters[5], use_relu=True, use_batch_norm=True, training_phase=train_phase)
        enc_c5_b = layers.conv2d_layer(ip_layer=enc_c5_a, name='enc_c5_b', num_filters=no_filters[5], use_relu=True, use_batch_norm=True, training_phase=train_phase)

        ########################
        # Decoder - Upsampling Path
        ########################
        # Upsample + 2x2 conv to half the no. of feature channels + SKIP connection (concate the conv. layers)
        # Level 5 - 1 upsampling layer + 1 conv op. + skip connection + 2x conv op.
        scale_val=2
        dec_up5 = layers.upsample_layer(ip_layer=enc_c5_b, method=self.method_val, scale_factor=scale_val)
        dec_dc5 = layers.conv2d_layer(ip_layer=dec_up5,name='dec_dc5', kernel_size=(fs_de,fs_de),num_filters=no_filters[4],use_relu=True, use_batch_norm=True, training_phase=train_phase)
        dec_cat_c5 = tf.concat((dec_dc5,enc_c4_b),axis=3,name='dec_cat_c5')

        #Level 4
        dec_c4_a = layers.conv2d_layer(ip_layer=dec_cat_c5,name='dec_c4_a', num_filters=no_filters[4], use_relu=True, use_batch_norm=True, training_phase=train_phase)
        dec_c4_b = layers.conv2d_layer(ip_layer=dec_c4_a,name='dec_c4_b', num_filters=no_filters[4], use_relu=True, use_batch_norm=True, training_phase=train_phase)
        dec_up4 = layers.upsample_layer(ip_layer=dec_c4_b, method=self.method_val, scale_factor=scale_val)
        dec_dc4 = layers.conv2d_layer(ip_layer=dec_up4,name='dec_dc4', kernel_size=(fs_de,fs_de),num_filters=no_filters[3],use_relu=True, use_batch_norm=True, training_phase=train_phase)
        dec_cat_c4 = tf.concat((dec_dc4,enc_c3_b),axis=3,name='dec_cat_c4')

        #Level 3
        dec_c3_a = layers.conv2d_layer(ip_layer=dec_cat_c4,name='dec_c3_a', num_filters=no_filters[3], use_relu=True, use_batch_norm=True, training_phase=train_phase)
        dec_c3_b = layers.conv2d_layer(ip_layer=dec_c3_a,name='dec_c3_b', num_filters=no_filters[3], use_relu=True, use_batch_norm=True, training_phase=train_phase)
        dec_up3 = layers.upsample_layer(ip_layer=dec_c3_b, method=self.method_val, scale_factor=scale_val)
        dec_dc3 = layers.conv2d_layer(ip_layer=dec_up3,name='dec_dc3', kernel_size=(fs_de,fs_de),num_filters=no_filters[2],use_relu=True, use_batch_norm=True, training_phase=train_phase)
        dec_cat_c3 = tf.concat((dec_dc3,enc_c2_b),axis=3,name='dec_cat_c3')

        #Level 2
        dec_c2_a = layers.conv2d_layer(ip_layer=dec_cat_c3,name='dec_c2_a', num_filters=no_filters[2], use_relu=True, use_batch_norm=True, training_phase=train_phase)
        dec_c2_b = layers.conv2d_layer(ip_layer=dec_c2_a,name='dec_c2_b', num_filters=no_filters[2], use_relu=True, use_batch_norm=True, training_phase=train_phase)
        dec_up2 = layers.upsample_layer(ip_layer=dec_c2_b, method=self.method_val, scale_factor=scale_val)
        dec_dc2 = layers.conv2d_layer(ip_layer=dec_up2,name='dec_dc2', kernel_size=(fs_de,fs_de),num_filters=no_filters[1],use_relu=True, use_batch_norm=True, training_phase=train_phase)
        dec_cat_c2 = tf.concat((dec_dc2,enc_c1_b),axis=3,name='dec_cat_c2')

        # Level 1 - multiple conv ops.
        dec_c1_a = layers.conv2d_layer(ip_layer=dec_cat_c2,name='dec_c1_a', num_filters=no_filters[1], use_relu=True, use_batch_norm=True, training_phase=train_phase)
        seg_c1_a = layers.conv2d_layer(ip_layer=dec_c1_a,name='seg_c1_a',num_filters=no_filters[1], use_relu=True, use_batch_norm=True, training_phase=train_phase)
        seg_c1_b = layers.conv2d_layer(ip_layer=seg_c1_a,name='seg_c1_b', num_filters=no_filters[1], use_relu=True, use_batch_norm=True, training_phase=train_phase)
        seg_c1_c = layers.conv2d_layer(ip_layer=seg_c1_b,name='seg_c1_c', num_filters=no_filters[1], use_relu=True, use_batch_norm=True, training_phase=train_phase)

        # Final output layer - Logits before softmax
        seg_fin_layer = layers.conv2d_layer(ip_layer=seg_c1_c,name='seg_fin_layer', num_filters=self.num_classes,use_relu=False, use_batch_norm=False, training_phase=train_phase)

        # Predict Class
        y_pred = tf.nn.softmax(seg_fin_layer)
        y_pred_cls = tf.argmax(y_pred,axis=3)

        ########################
        # Simple Cross Entropy (CE) between predicted labels and true labels - for only labelled data
        if(dsc_loss==1):
            # For dice score loss function
            #without background
            seg_cost = loss.dice_loss_without_backgrnd(logits=seg_fin_layer, labels=y_l_onehot)
            seg_cost_wgtce = seg_cost
            #with background
            #seg_cost = dice_loss_with_backgrnd(logits=seg_fin_layer, labels=y_l_onehot)
        else:
            # For Weighted CE loss function
            seg_cost = loss.pixel_wise_cross_entropy_loss_weighted_nn(logits=seg_fin_layer, labels=y_l_onehot, class_weights=class_weights)
            seg_cost_wgtce = seg_cost

        # get the var list for Segmentation Network
        seg_net_vars = []
        for v in tf.trainable_variables():
            var_name = v.name
            if 'enc_' in var_name: seg_net_vars.append(v)
            elif 'dec_' in var_name: seg_net_vars.append(v)
            elif 'seg_' in var_name: seg_net_vars.append(v)


        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):

            cost_a1=-lamda_l1_g*tf.reduce_mean(tf.abs(tf.layers.flatten(flow_vec))) + lamda_adv*tf.reduce_mean(g_cost)
            cost_a1_seg=cost_a1+lamda_dsc*tf.reduce_mean(seg_cost_wgtce)

            optimizer_l2_gen_seg = tf.train.AdamOptimizer(learning_rate=learn_rate_gen,beta1=beta1_val).minimize(cost_a1_seg, var_list=gen_net_vars)
            optimizer_l2_gen = tf.train.AdamOptimizer(learning_rate=learn_rate_gen,beta1=beta1_val).minimize(cost_a1, var_list=gen_net_vars)
            optimizer_l2_both_gen_unet = tf.train.AdamOptimizer(learning_rate=learn_rate_gen,beta1=beta1_val).minimize(cost_a1_seg, var_list=gen_net_vars+seg_net_vars)

            cost_a2=tf.reduce_mean(z_cost)
            optimizer_disc = tf.train.AdamOptimizer(learning_rate=learn_rate_disc,beta1=beta1_val).minimize(cost_a2, var_list=disc_net_vars)

            cost_a1_seg_loss=tf.reduce_mean(seg_cost)
            optimizer_unet_seg = tf.train.AdamOptimizer(learn_rate_seg).minimize(cost_a1_seg_loss,var_list=seg_net_vars)


        z_summary = tf.summary.scalar('z_cost', tf.reduce_mean(z_cost))
        g_summary = tf.summary.scalar('g_cost', tf.reduce_mean(g_cost))
        g_a1_summary = tf.summary.scalar('g_cost_a1', tf.reduce_mean(cost_a1))
        g_a1_seg_summary = tf.summary.scalar('g_cost_a1_seg', tf.reduce_mean(cost_a1_seg))
        flow_summary=tf.summary.scalar('flow_vec_mean',tf.reduce_mean(tf.abs(tf.layers.flatten(flow_vec))))
        seg_summary = tf.summary.scalar('seg_cost', tf.reduce_mean(seg_cost))

        train_summary = tf.summary.merge([z_summary,g_summary,flow_summary,g_a1_summary,g_a1_seg_summary])

        # For dice score summary
        rv_dice = tf.placeholder(tf.float32, shape=[], name='rv_dice')
        rv_dice_summary = tf.summary.scalar('rv_val_dice', rv_dice)
        myo_dice = tf.placeholder(tf.float32, shape=[], name='myo_dice')
        myo_dice_summary = tf.summary.scalar('myo_val_dice', myo_dice)
        lv_dice = tf.placeholder(tf.float32, shape=[], name='lv_dice')
        lv_dice_summary = tf.summary.scalar('lv_val_dice', lv_dice)

        mean_dice = tf.placeholder(tf.float32, shape=[], name='mean_dice')
        mean_dice_summary = tf.summary.scalar('mean_val_dice', mean_dice)

        val_dsc_summary = tf.summary.merge([mean_dice_summary,rv_dice_summary,myo_dice_summary,lv_dice_summary])

        val_totalc = tf.placeholder(tf.float32, shape=[], name='val_totalc')
        val_totalc_sum= tf.summary.scalar('val_totalc_', val_totalc)
        val_summary = tf.summary.merge([val_totalc_sum])

        return {'x': x, 'z':z, 'y_l':y_l, 'train_phase':train_phase, 'seg_cost': seg_cost,\
        'x_l':x_l,'x_unl':x_unl,'select_mask': select_mask,'z_cost':cost_a2,'g_cost':cost_a1,'g_net_cost':cost_a1_seg,\
        'y_pred' : y_pred, 'y_pred_cls': y_pred_cls,\
        'train_summary':train_summary,'y_trans':y_trans,'z_class':z_class,'z_pred':z_pred,'z_pred_cls':z_pred_cls,\
        'optimizer_disc':optimizer_disc,'optimizer_l2_gen':optimizer_l2_gen,'optimizer_unet_seg' :optimizer_unet_seg, \
        'optimizer_l2_gen_seg':optimizer_l2_gen_seg,'seg_summary':seg_summary,\
        'optimizer_l2_both_gen_unet':optimizer_l2_both_gen_unet,\
        'flow_vec':flow_vec,'rv_dice':rv_dice,'myo_dice':myo_dice,'lv_dice':lv_dice,'mean_dice':mean_dice,'val_dsc_summary':val_dsc_summary,\
        'val_totalc':val_totalc,'val_summary':val_summary}


    def intensity_transform_cgan_unet(self,learn_rate_gen=0.0001,learn_rate_disc=0.0001,z_lat_dim=100,lat_dim=128,beta1_val=0.9,\
                     gan_type='gan',ra_en=True,learn_rate_seg=0.001,dsc_loss=1,en_1hot=0,lamda_dsc=1,lamda_adv=1,lamda_l1_i=0.001):

        no_filters=[1, 16, 32, 64, 128, 256]

        if(self.num_classes==2):
            class_weights = tf.constant([[0.05, 0.95]],name='class_weights')
        elif(self.num_classes==3):
            class_weights = tf.constant([[0.05, 0.5, 0.45]],name='class_weights')
        elif(self.num_classes==4):
            class_weights = tf.constant([[0.1, 0.3, 0.3, 0.3]],name='class_weights')

        std_init=0.01
        SEED=1
        acti='xavier'
        intl=tf.truncated_normal_initializer(stddev=std_init, seed=SEED)

        hid_dim=int(self.img_size_x*self.img_size_y*no_filters[4]/(32*32))
        latent_dim=lat_dim # 128 #or 512 / 128

        z_hid_dim=int(self.img_size_x*self.img_size_y*no_filters[4]/(32*32))
        dim_x = int(self.img_size_x/32)

        # Generator - FCN variables
        gen_c1_weights = tf.get_variable(name="gen_c1_weights",shape=[z_lat_dim,z_hid_dim], initializer=intl)
        gen_c1_biases = tf.get_variable(name="gen_c1_biases",shape=[z_hid_dim], initializer=tf.constant_initializer(value=0))

        # Discriminator - FCN variables
        fcn_c1_weights = tf.get_variable(name="fcn_c1_weights",shape=[hid_dim, latent_dim], initializer=intl)
        fcn_c1_biases = tf.get_variable(name="fcn_c1_biases",shape=[latent_dim], initializer=tf.constant_initializer(value=0))
        fcn_c2_weights = tf.get_variable(name="fcn_c2_weights",shape=[latent_dim, latent_dim], initializer=intl)
        fcn_c2_biases = tf.get_variable(name="fcn_c2_biases",shape=[latent_dim], initializer=tf.constant_initializer(value=0))
        fcn_c3_weights = tf.get_variable(name="fcn_c3_weights",shape=[latent_dim,1], initializer=intl)
        fcn_c3_biases = tf.get_variable(name="fcn_c3_biases",shape=[1], initializer=tf.constant_initializer(value=0))

        num_channels=no_filters[0]
        # Placeholders
        # input to the network
        z = tf.placeholder(tf.float32, shape=[self.batch_size, z_lat_dim], name='z')
        x = tf.placeholder(tf.float32, shape=[None, self.img_size_x, self.img_size_y, num_channels], name='x')
        x_unl = tf.placeholder(tf.float32, shape=[None, self.img_size_x, self.img_size_y, num_channels], name='x_unl')
        if(en_1hot==1):
            y_l = tf.placeholder(tf.float32, shape=[None, self.img_size_x, self.img_size_y,self.num_classes], name='y_l')
        else:
            y_l = tf.placeholder(tf.int32, shape=[None, self.img_size_x, self.img_size_y], name='y_l')
        select_mask = tf.placeholder(tf.bool, name='select_mask')
        train_phase = tf.placeholder(tf.bool, name='train_phase')
        if(en_1hot==0):
            y_l_onehot=tf.one_hot(y_l,depth=self.num_classes)
        else:
            y_l_onehot=y_l

        ############################################
        # Generator Network
        ############################################
        # Dense layer + Reshape to down sampled image dimensions
        gen_fcn_c1 = tf.matmul(z, gen_c1_weights) + gen_c1_biases
        gen_fcn_relu_c1 = tf.nn.relu(gen_fcn_c1)
        gen_fcn_reshaped = tf.reshape(gen_fcn_relu_c1 ,[-1,dim_x,dim_x,no_filters[4]])

        # Level 5 - Upsample + Conv. op
        fs_de=2
        scale_val=2
        gen_up5 = layers.upsample_layer(ip_layer=gen_fcn_reshaped, method=self.method_val, scale_factor=scale_val)
        gen_c5 = layers.conv2d_layer(ip_layer=gen_up5,name='gen_c5', kernel_size=(fs_de,fs_de),num_filters=no_filters[4],use_relu=True, use_batch_norm=True, training_phase=train_phase)

        # Level 4
        gen_up4 = layers.upsample_layer(ip_layer=gen_c5, method=self.method_val, scale_factor=scale_val)
        gen_c4 = layers.conv2d_layer(ip_layer=gen_up4,name='gen_c4', kernel_size=(fs_de,fs_de),num_filters=no_filters[3],use_relu=True, use_batch_norm=True, training_phase=train_phase)

        # Level 3
        gen_up3 = layers.upsample_layer(ip_layer=gen_c4, method=self.method_val, scale_factor=scale_val)
        gen_c3 = layers.conv2d_layer(ip_layer=gen_up3,name='gen_c3', kernel_size=(fs_de,fs_de),num_filters=no_filters[2],use_relu=True, use_batch_norm=True, training_phase=train_phase)

        # Level 2
        gen_up2 = layers.upsample_layer(ip_layer=gen_c3, method=self.method_val, scale_factor=scale_val)
        gen_c2 = layers.conv2d_layer(ip_layer=gen_up2,name='gen_c2', kernel_size=(fs_de,fs_de),num_filters=no_filters[1],use_relu=True, use_batch_norm=True, training_phase=train_phase)

        # Level 1
        gen_up1 = layers.upsample_layer(ip_layer=gen_c2, method=self.method_val, scale_factor=scale_val)
        gen_c1 = layers.conv2d_layer(ip_layer=gen_up1,name='gen_c1', kernel_size=(fs_de,fs_de),num_filters=no_filters[1],use_relu=False, use_batch_norm=False, training_phase=train_phase)


        # Conv. ops on input image
        conv_1a = layers.conv2d_layer(ip_layer=x,name='conv_1a',num_filters=no_filters[1], use_relu=True, use_batch_norm=True, training_phase=train_phase)
        conv_1b = layers.conv2d_layer(ip_layer=conv_1a,name='conv_1b',num_filters=no_filters[1], use_relu=True, use_batch_norm=True, training_phase=train_phase)

        # Concatenate features obtained by conv. ops on image and on 'z'
        gen_cat = tf.concat((gen_c1, conv_1b), axis=3)

        # More Conv. ops on concatenated feature maps
        conv_1c = layers.conv2d_layer(ip_layer=gen_cat,name='conv_1c',num_filters=no_filters[1], use_relu=True, use_batch_norm=True, training_phase=train_phase)
        conv_1d = layers.conv2d_layer(ip_layer=conv_1c,name='conv_1d',num_filters=no_filters[1], use_relu=True, use_batch_norm=True, training_phase=train_phase)
        #conv_1e = layers.conv2d_layer(ip_layer=conv_1d,name='conv_1e',num_filters=2, use_relu=False, use_batch_norm=False, training_phase=train_phase)

        # intensity maps to add to transformed image
        int_c1 = layers.conv2d_layer(ip_layer=conv_1d, name='int_c1', kernel_size=(1, 1), num_filters=1,use_bias=False, use_relu=False, use_batch_norm=False, training_phase=train_phase)

        # tanh activation function to restrict the values of delta I (additive transform)
        int_c1=tf.nn.tanh(int_c1)

        # add intensity transform (int_c1) to the input image to yield intensity transformed image
        y_int = tf.add(x,int_c1)

    ############################################
        #Discriminator Network
    ############################################

        cat_disc_c1=tf.concat((y_int,x_unl),axis=0,name='cat_disc_c1')
        # Choose between concate or true+gen images or gen images
        cat_disc_c1 = tf.cond(select_mask,lambda : cat_disc_c1, lambda :y_int)

        # DISC Net Architecutre
        disc_c1 = layers.conv2d_layer(ip_layer=cat_disc_c1, name='disc_c1', num_filters=no_filters[1],kernel_size=(5, 5),strides=(2, 2),use_relu=False, use_batch_norm=True, training_phase=train_phase)
        disc_c1 = tf.nn.leaky_relu(disc_c1, alpha=0.2)

        disc_c2 = layers.conv2d_layer(ip_layer=disc_c1, name='disc_c2', num_filters=no_filters[2],kernel_size=(5, 5),strides=(2, 2),use_relu=False, use_batch_norm=True, training_phase=train_phase)
        disc_c2 = tf.nn.leaky_relu(disc_c2, alpha=0.2)

        disc_c3 = layers.conv2d_layer(ip_layer=disc_c2, name='disc_c3', num_filters=no_filters[3],kernel_size=(5, 5),strides=(2, 2),use_relu=False, use_batch_norm=True, training_phase=train_phase)
        disc_c3 = tf.nn.leaky_relu(disc_c3, alpha=0.2)

        disc_c4 = layers.conv2d_layer(ip_layer=disc_c3, name='disc_c4', num_filters=no_filters[4],kernel_size=(5, 5),strides=(2, 2),use_relu=False, use_batch_norm=True, training_phase=train_phase)
        disc_c4 = tf.nn.leaky_relu(disc_c4, alpha=0.2)

        disc_c5 = layers.conv2d_layer(ip_layer=disc_c4, name='disc_c5', num_filters=no_filters[4],kernel_size=(5, 5),strides=(2, 2),use_relu=False, use_batch_norm=True, training_phase=train_phase)
        disc_c5_pool = tf.nn.leaky_relu(disc_c5, alpha=0.2)

        # Flat conv for FCN
        flat_conv = tf.contrib.layers.flatten(disc_c5_pool)

        # FCN + Relu - x2
        z_fcn_c1 = tf.matmul(flat_conv, fcn_c1_weights) + fcn_c1_biases
        z_fcn_relu_c1 = tf.nn.leaky_relu(z_fcn_c1)

        z_fcn_c2 = tf.matmul(z_fcn_relu_c1, fcn_c2_weights) + fcn_c2_biases
        z_fcn_relu_c2 = tf.nn.leaky_relu(z_fcn_c2)

        # 1 fully connected layer to determine input images into real / fake categories
        z_class = tf.matmul(z_fcn_relu_c2, fcn_c3_weights) + fcn_c3_biases

        z_pred=z_class
        z_pred_cls=z_pred

        fake_indices=np.arange(0,self.batch_size,dtype=np.int32)
        fake = tf.gather(z_class,fake_indices)
        real_indices=np.arange(self.batch_size,2*self.batch_size,dtype=np.int32)
        real = tf.gather(z_class,real_indices)

        # Discriminator loss
        z_cost=self.discriminator_loss(Ra=ra_en, loss_func=gan_type, real=real, fake=fake)
        # Generator loss
        g_cost=self.generator_loss(Ra=ra_en, loss_func=gan_type, real=real, fake=fake)

        # divide the var list into Generator Network and Discriminator Net
        gen_net_vars = []
        disc_net_vars = []
        for v in tf.trainable_variables():
            var_name = v.name
            if 'disc_' in var_name: disc_net_vars.append(v)
            elif 'fcn_' in var_name: disc_net_vars.append(v)
            elif 'gen_' in var_name: gen_net_vars.append(v)
            elif 'conv_' in var_name: gen_net_vars.append(v)
            elif('int_' in var_name): gen_net_vars.append(v)

        ############################################
        # U-Net Network
        ############################################
        # Encoder - Downsampling Path
        ############################################
        # 2x 3x3 conv and 1 maxpool
        # Level 1
        enc_c1_a = layers.conv2d_layer(ip_layer=x, name='enc_c1_a', num_filters=no_filters[1], use_relu=True, use_batch_norm=True, training_phase=train_phase)
        enc_c1_b = layers.conv2d_layer(ip_layer=enc_c1_a, name='enc_c1_b', num_filters=no_filters[1], use_relu=True, use_batch_norm=True, training_phase=train_phase)
        enc_c1_pool = layers.max_pool_layer2d(enc_c1_b, kernel_size=(2, 2), strides=(2, 2), padding="SAME",name='enc_c1_pool')

        # Level 2
        enc_c2_a = layers.conv2d_layer(ip_layer=enc_c1_pool, name='enc_c2_a', num_filters=no_filters[2], use_relu=True, use_batch_norm=True, training_phase=train_phase)
        enc_c2_b = layers.conv2d_layer(ip_layer=enc_c2_a, name='enc_c2_b', num_filters=no_filters[2], use_relu=True, use_batch_norm=True, training_phase=train_phase)
        enc_c2_pool = layers.max_pool_layer2d(enc_c2_b, kernel_size=(2, 2), strides=(2, 2), padding="SAME",name='enc_c2_pool')

        # Level 3
        enc_c3_a = layers.conv2d_layer(ip_layer=enc_c2_pool, name='enc_c3_a', num_filters=no_filters[3],use_relu=True, use_batch_norm=True, training_phase=train_phase)
        enc_c3_b = layers.conv2d_layer(ip_layer=enc_c3_a, name='enc_c3_b', num_filters=no_filters[3], use_relu=True,use_batch_norm=True, training_phase=train_phase)
        enc_c3_pool = layers.max_pool_layer2d(enc_c3_b, kernel_size=(2, 2), strides=(2, 2), padding="SAME",name='enc_c3_pool')

        # Level 4
        enc_c4_a = layers.conv2d_layer(ip_layer=enc_c3_pool, name='enc_c4_a', num_filters=no_filters[4],use_relu=True, use_batch_norm=True, training_phase=train_phase)
        enc_c4_b = layers.conv2d_layer(ip_layer=enc_c4_a, name='enc_c4_b', num_filters=no_filters[4], use_relu=True,use_batch_norm=True, training_phase=train_phase)
        enc_c4_pool = layers.max_pool_layer2d(enc_c4_b, kernel_size=(2, 2), strides=(2, 2), padding="SAME",name='enc_c4_pool')

        # Level 5 - 2x Conv
        enc_c5_a = layers.conv2d_layer(ip_layer=enc_c4_pool, name='enc_c5_a', num_filters=no_filters[5],use_relu=True, use_batch_norm=True, training_phase=train_phase)
        enc_c5_b = layers.conv2d_layer(ip_layer=enc_c5_a, name='enc_c5_b', num_filters=no_filters[5], use_relu=True,use_batch_norm=True, training_phase=train_phase)

        ########################
        # Decoder - Upsampling Path
        ########################
        # Upsample + 2x2 conv to half the no. of feature channels + SKIP connection (concate the conv. layers)
        # Level 5 - 1 upsampling layer + 1 conv op. + skip connection + 2x conv op.
        scale_val = 2
        dec_up5 = layers.upsample_layer(ip_layer=enc_c5_b, method=self.method_val, scale_factor=scale_val)
        dec_dc5 = layers.conv2d_layer(ip_layer=dec_up5, name='dec_dc5', kernel_size=(fs_de, fs_de),num_filters=no_filters[4], use_relu=True, use_batch_norm=True,training_phase=train_phase)
        dec_cat_c5 = tf.concat((dec_dc5, enc_c4_b), axis=3, name='dec_cat_c5')

        # Level 4
        dec_c4_a = layers.conv2d_layer(ip_layer=dec_cat_c5, name='dec_c4_a', num_filters=no_filters[4],use_relu=True, use_batch_norm=True, training_phase=train_phase)
        dec_c4_b = layers.conv2d_layer(ip_layer=dec_c4_a, name='dec_c4_b', num_filters=no_filters[4], use_relu=True,use_batch_norm=True, training_phase=train_phase)
        dec_up4 = layers.upsample_layer(ip_layer=dec_c4_b, method=self.method_val, scale_factor=scale_val)
        dec_dc4 = layers.conv2d_layer(ip_layer=dec_up4, name='dec_dc4', kernel_size=(fs_de, fs_de),num_filters=no_filters[3], use_relu=True, use_batch_norm=True,training_phase=train_phase)
        dec_cat_c4 = tf.concat((dec_dc4, enc_c3_b), axis=3, name='dec_cat_c4')

        # Level 3 -
        dec_c3_a = layers.conv2d_layer(ip_layer=dec_cat_c4, name='dec_c3_a', num_filters=no_filters[3],use_relu=True, use_batch_norm=True, training_phase=train_phase)
        dec_c3_b = layers.conv2d_layer(ip_layer=dec_c3_a, name='dec_c3_b', num_filters=no_filters[3], use_relu=True,use_batch_norm=True, training_phase=train_phase)
        dec_up3 = layers.upsample_layer(ip_layer=dec_c3_b, method=self.method_val, scale_factor=scale_val)
        dec_dc3 = layers.conv2d_layer(ip_layer=dec_up3, name='dec_dc3', kernel_size=(fs_de, fs_de),num_filters=no_filters[2], use_relu=True, use_batch_norm=True,training_phase=train_phase)
        dec_cat_c3 = tf.concat((dec_dc3, enc_c2_b), axis=3, name='dec_cat_c3')

        # Level 2 -
        dec_c2_a = layers.conv2d_layer(ip_layer=dec_cat_c3, name='dec_c2_a', num_filters=no_filters[2],use_relu=True, use_batch_norm=True, training_phase=train_phase)
        dec_c2_b = layers.conv2d_layer(ip_layer=dec_c2_a, name='dec_c2_b', num_filters=no_filters[2], use_relu=True,use_batch_norm=True, training_phase=train_phase)
        dec_up2 = layers.upsample_layer(ip_layer=dec_c2_b, method=self.method_val, scale_factor=scale_val)
        dec_dc2 = layers.conv2d_layer(ip_layer=dec_up2, name='dec_dc2', kernel_size=(fs_de, fs_de),num_filters=no_filters[1], use_relu=True, use_batch_norm=True,training_phase=train_phase)
        dec_cat_c2 = tf.concat((dec_dc2, enc_c1_b), axis=3, name='dec_cat_c2')

        # Level 1 - multiple conv ops.
        dec_c1_a = layers.conv2d_layer(ip_layer=dec_cat_c2, name='dec_c1_a', num_filters=no_filters[1],use_relu=True, use_batch_norm=True, training_phase=train_phase)
        seg_c1_a = layers.conv2d_layer(ip_layer=dec_c1_a, name='seg_c1_a', num_filters=no_filters[1], use_relu=True,use_batch_norm=True, training_phase=train_phase)
        seg_c1_b = layers.conv2d_layer(ip_layer=seg_c1_a, name='seg_c1_b', num_filters=no_filters[1], use_relu=True,use_batch_norm=True, training_phase=train_phase)
        seg_c1_c = layers.conv2d_layer(ip_layer=seg_c1_b, name='seg_c1_c', num_filters=no_filters[1], use_relu=True,use_batch_norm=True, training_phase=train_phase)

        # Final output layer - Logits before softmax
        seg_fin_layer = layers.conv2d_layer(ip_layer=seg_c1_c, name='seg_fin_layer', num_filters=self.num_classes,use_relu=False, use_batch_norm=False, training_phase=train_phase)

        # Predict Class
        y_pred = tf.nn.softmax(seg_fin_layer)
        y_pred_cls = tf.argmax(y_pred, axis=3)

        ########################
        # Simple Cross Entropy (CE) between predicted labels and true labels - for only labelled data
        if(dsc_loss==1):
            # For dice score loss function
            #without background
            seg_cost = loss.dice_loss_without_backgrnd(logits=seg_fin_layer, labels=y_l_onehot)
            seg_cost_wgtce=seg_cost
            #with background
            #seg_cost = dice_loss_with_backgrnd(logits=seg_fin_layer, labels=y_l_onehot)
        else:
            # For Weighted CE loss function
            seg_cost = loss.pixel_wise_cross_entropy_loss_weighted_nn(logits=seg_fin_layer, labels=y_l_onehot, class_weights=class_weights)
            seg_cost_wgtce=seg_cost

        # get the var list for Segmentation Network
        seg_net_vars = []
        for v in tf.trainable_variables():
            var_name = v.name
            if 'enc_' in var_name: seg_net_vars.append(v)
            elif 'dec_' in var_name: seg_net_vars.append(v)
            elif 'seg_' in var_name: seg_net_vars.append(v)


        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            cost_a1=-lamda_l1_i*tf.reduce_mean(tf.abs(int_c1)) + lamda_adv*tf.reduce_mean(g_cost)
            cost_a1_seg=cost_a1+lamda_dsc*tf.reduce_mean(seg_cost_wgtce)

            optimizer_l2_gen = tf.train.AdamOptimizer(learning_rate=learn_rate_gen,beta1=beta1_val).minimize(cost_a1, var_list=gen_net_vars)
            optimizer_l2_gen_seg = tf.train.AdamOptimizer(learning_rate=learn_rate_gen,beta1=beta1_val).minimize(cost_a1_seg, var_list=gen_net_vars)
            optimizer_l2_both_gen_unet = tf.train.AdamOptimizer(learning_rate=learn_rate_gen,beta1=beta1_val).minimize(cost_a1_seg, var_list=gen_net_vars+seg_net_vars)

            cost_a2=tf.reduce_mean(z_cost)
            optimizer_disc = tf.train.AdamOptimizer(learning_rate=learn_rate_disc,beta1=beta1_val).minimize(cost_a2, var_list=disc_net_vars)

            cost_a1_seg_loss=tf.reduce_mean(seg_cost)
            optimizer_unet_seg = tf.train.AdamOptimizer(learn_rate_seg).minimize(cost_a1_seg_loss,var_list=seg_net_vars)
            

        z_summary = tf.summary.scalar('z_cost', tf.reduce_mean(z_cost))
        g_summary = tf.summary.scalar('g_cost', tf.reduce_mean(g_cost))
        g_a1_summary = tf.summary.scalar('g_cost_a1', tf.reduce_mean(cost_a1))
        g_a1_seg_summary = tf.summary.scalar('g_cost_a1_seg', tf.reduce_mean(cost_a1_seg))
        int_c1_summary=tf.summary.scalar('int_c1',tf.reduce_mean(tf.abs(tf.layers.flatten(int_c1))))
        seg_summary = tf.summary.scalar('seg_cost', tf.reduce_mean(seg_cost))

        #train_summary = tf.summary.merge([z_summary,g_summary,flow_summary])
        train_summary = tf.summary.merge([z_summary,g_summary,int_c1_summary,g_a1_summary,g_a1_seg_summary])

        # For dice score summary
        rv_dice = tf.placeholder(tf.float32, shape=[], name='rv_dice')
        rv_dice_summary = tf.summary.scalar('rv_val_dice', rv_dice)
        myo_dice = tf.placeholder(tf.float32, shape=[], name='myo_dice')
        myo_dice_summary = tf.summary.scalar('myo_val_dice', myo_dice)
        lv_dice = tf.placeholder(tf.float32, shape=[], name='lv_dice')
        lv_dice_summary = tf.summary.scalar('lv_val_dice', lv_dice)

        mean_dice = tf.placeholder(tf.float32, shape=[], name='mean_dice')
        mean_dice_summary = tf.summary.scalar('mean_val_dice', mean_dice)

        val_dsc_summary = tf.summary.merge([mean_dice_summary,rv_dice_summary,myo_dice_summary,lv_dice_summary])
        val_totalc = tf.placeholder(tf.float32, shape=[], name='val_totalc')
        val_totalc_sum= tf.summary.scalar('val_totalc_', val_totalc)
        val_summary = tf.summary.merge([val_totalc_sum])


        return {'x': x, 'z':z, 'y_l':y_l, 'train_phase':train_phase, 'seg_cost': seg_cost,\
                'x_unl':x_unl,'select_mask': select_mask,'z_cost':cost_a2,'g_cost':cost_a1,'g_net_cost':cost_a1_seg,\
                'y_pred' : y_pred, 'y_pred_cls': y_pred_cls,\
                'train_summary':train_summary,'z_class':z_class,'z_pred':z_pred,'z_pred_cls':z_pred_cls,\
                'optimizer_disc':optimizer_disc,'optimizer_l2_gen':optimizer_l2_gen,'optimizer_unet_seg' :optimizer_unet_seg, \
                'optimizer_l2_gen_seg':optimizer_l2_gen_seg,'seg_summary':seg_summary,\
                'optimizer_l2_both_gen_unet':optimizer_l2_both_gen_unet,\
                'int_c1':int_c1,'y_int':y_int,\
                'rv_dice':rv_dice,'myo_dice':myo_dice,'lv_dice':lv_dice,'mean_dice':mean_dice,'val_dsc_summary':val_dsc_summary,\
                'val_totalc':val_totalc,'val_summary':val_summary}

