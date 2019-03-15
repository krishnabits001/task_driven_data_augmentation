import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score
import nibabel as nib

#to make directories
import pathlib
from skimage import transform

from scipy.ndimage import morphology
from array2gif import write_gif

class f1_utilsObj:
    def __init__(self,cfg,dt):
        #print('f1 utils init')
        self.img_size_x=cfg.img_size_x
        self.img_size_y=cfg.img_size_y
        self.batch_size=cfg.batch_size
        self.num_classes=cfg.num_classes
        self.num_channels=cfg.num_channels
        self.method_val = cfg.method_val
        self.target_resolution=cfg.target_resolution
        self.data_path_tr=cfg.data_path_tr
        self.dt=dt

    def surfd(self,input1, input2, sampling=1, connectivity=1):
        '''
        function to compute the surface distance
        input params:
            input1: predicted segmentation mask
            input2: ground truth mask
            sampling: default value
            connectivity: default value
        returns:
            sds : surface distance
        '''
        input_1 = np.atleast_1d(input1.astype(np.bool))
        input_2 = np.atleast_1d(input2.astype(np.bool))
        conn = morphology.generate_binary_structure(input_1.ndim, connectivity)

        #binary erosion on input1
        y=morphology.binary_erosion(input_1, conn)
        y=y.astype(np.float32)
        x=input_1.astype(np.float32)
        S=x-y

        #binary erosion on input2
        y=morphology.binary_erosion(input_2, conn)
        y=y.astype(np.float32)
        x=input_2.astype(np.float32)
        Sprime=x-y

        S=S.astype(np.bool)
        Sprime=Sprime.astype(np.bool)

        dta = morphology.distance_transform_edt(~S,sampling)
        dtb = morphology.distance_transform_edt(~Sprime,sampling)

        sds = np.concatenate([np.ravel(dta[Sprime!=0]), np.ravel(dtb[S!=0])])

        return sds

    def calc_pred_sf_mask(self, sess, ae, labeled_data_imgs, axis_no=2):
        """
        To compute the predicted segmentation for an input stack of 2D slices
        input params:
            sess: current session
            ae: graph name
            labeled_data_imgs: input 3D volume
            axis_no:
        returns:
            mergedlist_y_pred: predicted segmentation masks of all 2D slices
        """
        total_slices = labeled_data_imgs.shape[axis_no]
        for slice_no in range(total_slices):
            img_test_slice = np.reshape(labeled_data_imgs[:, :, slice_no], (1, self.img_size_x, self.img_size_y, 1))

            seg_pred = sess.run(ae['y_pred'], feed_dict={ae['x']: img_test_slice, ae['train_phase']: False})

            # Merging predicted labels of slices(2D) of test image into one volume(3D) of predicted labels
            if (slice_no == 0):
                mergedlist_y_pred = np.reshape(seg_pred, (1,self.img_size_x, self.img_size_y, self.num_classes))
            else:
                seg_pred_final = np.reshape(seg_pred, (1,self.img_size_x, self.img_size_y, self.num_classes))
                mergedlist_y_pred = np.concatenate((mergedlist_y_pred, seg_pred_final), axis=0)

        return mergedlist_y_pred

    def calc_pred_sf_mask_full(self, sess, ae, labeled_data_imgs):
        '''
        To compute the predicted segmentation for an input 3D volume
        input params:
            sess: current session
            ae: graph name
            labeled_data_imgs: input 3D volume
        returns:
            seg_pred: predicted segmentation mask of 3D volume
        '''
        test_data = labeled_data_imgs
        seg_pred = sess.run(ae['y_pred'], feed_dict={ae['x']: test_data, ae['train_phase']: False})

        return seg_pred

    def reshape_img_and_f1_score(self, predicted_img_arr, gt_mask, pixel_size):
        '''
        To reshape image into the target resolution and then compute the f1 score w.r.t ground truth mask
        input params:
            predicted_img_arr: predicted segmentation mask that is computed over the re-sampled and cropped input image
            gt_mask: ground truth mask in native image resolution
            pixel_size: native image resolution
        returns:
            predictions_mask: predictions mask in native resolution (re-sampled and cropped/zeros append as per size requirements)
            f1_val: f1 score over predicted segmentation masks vs ground truth
        '''
        nx,ny= self.img_size_x,self.img_size_y

        scale_vector = (pixel_size[0] / self.target_resolution[0], pixel_size[1] / self.target_resolution[1])
        mask_rescaled = transform.rescale(gt_mask[:, :, 0], scale_vector, order=0, preserve_range=True, mode='constant')
        x, y = mask_rescaled.shape[0],mask_rescaled.shape[1]

        x_s = (x - nx) // 2
        y_s = (y - ny) // 2
        x_c = (nx - x) // 2
        y_c = (ny - y) // 2

        total_slices = predicted_img_arr.shape[0]
        predictions_mask = np.zeros((gt_mask.shape[0],gt_mask.shape[1],total_slices))
        for slice_no in range(total_slices):
            # ASSEMBLE BACK THE SLICES
            slice_predictions = np.zeros((x,y,self.num_classes))
            predicted_img=predicted_img_arr[slice_no,:,:,:]
            # insert cropped region into original image again
            if x > nx and y > ny:
                slice_predictions[x_s:x_s+nx, y_s:y_s+ny,:] = predicted_img
            else:
                if x <= nx and y > ny:
                    slice_predictions[:, y_s:y_s+ny,:] = predicted_img[x_c:x_c+ x, :,:]
                elif x > nx and y <= ny:
                    slice_predictions[x_s:x_s + nx, :,:] = predicted_img[:, y_c:y_c + y,:]
                else:
                    slice_predictions[:, :,:] = predicted_img[x_c:x_c+ x, y_c:y_c + y,:]

            # RESCALING ON THE LOGITS
            prediction = transform.resize(slice_predictions,
                                              (gt_mask.shape[0], gt_mask.shape[1], self.num_classes),
                                              order=1,
                                              preserve_range=True,
                                              mode='constant')
            #print("b",prediction.shape)
            prediction = np.uint16(np.argmax(prediction, axis=-1))

            predictions_mask[:,:,slice_no]=prediction

        #Calculate F1 score
        #y_pred= predictions_mask.flatten()
        #y_true= gt_mask.flatten()
        #f1_val= f1_score(y_true, y_pred, average=None)
        f1_val = self.calc_f1_score(predictions_mask,gt_mask)

        return predictions_mask,f1_val

    def calc_f1_score(self,predictions_mask,gt_mask):
        '''
        to compute f1/dice score
        input params:
            predictions_arr: predicted segmentation mask
            mask: ground truth mask
        returns:
            f1_val: f1/dice score
        '''
        y_pred= predictions_mask.flatten()
        y_true= gt_mask.flatten()

        f1_val= f1_score(y_true, y_pred, average=None)

        return f1_val


    def pred_segs_acdc_test_subjs(self, sess,ae, save_dir,orig_img_dt,test_list,struct_name,print_assd_hd_scores=0):
        '''
        To estimate the segmentation masks of test images and compute their f1 score and plot the predicted segmentations.
        input params:
            sess: current session
            ae: current model graph
            save_dir: save directory for the inference of test images
            orig_img_dt: dataloader of acdc data
            test_list: list of patient test ids
            struct_name: list of structures to segment. Here its Right ventricle (RV), myocardium (MYO), left ventricle (LV) in the heart MRI.
        returns:
            None
        '''
        count=0
        # Load each test image
        for test_id in test_list:
            test_id_l=[test_id]

            #load image,label pairs and process it to chosen resolution and dimensions
            img_sys,label_sys,pixel_size,affine_tst= orig_img_dt(test_id_l,ret_affine=1)
            cropped_img_sys,cropped_mask_sys = self.dt.preprocess_data(img_sys, label_sys, pixel_size)

            # Make directory for the test image with id number
            seg_model_dir=str(save_dir)+'pred_segs/'+str(test_id)+'/'
            pathlib.Path(seg_model_dir).mkdir(parents=True, exist_ok=True)

            # Calc dice score and predicted segmentation & store in a txt file
            pred_sf_mask = self.calc_pred_sf_mask(sess, ae, cropped_img_sys, axis_no=2)
            re_pred_mask_sys,f1_val = self.reshape_img_and_f1_score(pred_sf_mask, label_sys, pixel_size)
            #print("mean f1_val", f1_val)
            savefile_name = str(seg_model_dir)+'mean_f1_dice_coeff_test_id_'+str(test_id)+'.txt'
            np.savetxt(savefile_name, f1_val, fmt='%s')

            # Save the segmentation in nrrd files & plot some sample images
            self.plot_predicted_seg_ss(img_sys,label_sys,re_pred_mask_sys,seg_model_dir,test_id)

            #save the nifti segmentation file
            array_img = nib.Nifti1Image(re_pred_mask_sys.astype(np.int16), affine_tst)
            pred_filename = str(seg_model_dir)+'pred_seg_id_'+str(test_id)+'.nii.gz'
            nib.save(array_img, pred_filename)

            dsc_tmp=np.reshape(f1_val[1:self.num_classes], (1, self.num_classes - 1))

            if(print_assd_hd_scores==1):
                assd_list=[]
                hd_list=[]
                for index in range(1,self.num_classes):
                    surface_distance = self.surfd((re_pred_mask_sys==index), (label_sys==index))
                    msd = surface_distance.mean()
                    hd=surface_distance.max()
                    assd_list.append(msd)
                    hd_list.append(hd)
                filename_msd=str(seg_model_dir)+'assd_test_id_'+str(test_id)+'.txt'
                filename_hd=str(seg_model_dir)+'hd_test_id_'+str(test_id)+'.txt'
                np.savetxt(filename_msd,assd_list,fmt='%s')
                np.savetxt(filename_hd,hd_list,fmt='%s')

                assd_tmp=np.reshape(np.asarray(assd_list),(1,self.num_classes-1))
                hd_tmp=np.reshape(np.asarray(hd_list),(1,self.num_classes-1))

            if(count==0):
                dsc_all=dsc_tmp
                if(print_assd_hd_scores==1):
                    assd_all=assd_tmp
                    hd_all=hd_tmp
                count=1
            else:
                dsc_all=np.concatenate((dsc_all, dsc_tmp))
                if(print_assd_hd_scores==1):
                    assd_all=np.concatenate((assd_all, assd_tmp))
                    hd_all=np.concatenate((hd_all, hd_tmp))

        #for DSC
        val_list=[]
        val_list_mean=[]
        for i in range(0,self.num_classes-1):
            dsc=dsc_all[:,i]
            #DSC
            #val_list.append(round(np.mean(dsc), 3))
            val_list.append(round(np.median(dsc), 3))
            val_list.append(round(np.std(dsc), 3))
            val_list_mean.append(round(np.mean(dsc), 3))
            filename_save=str(save_dir)+'pred_segs/'+str(struct_name[i])+'_20subjs_dsc.txt'
            np.savetxt(filename_save,dsc,fmt='%s')
        filename_save=str(save_dir)+'pred_segs/'+'median_std_dsc.txt'
        np.savetxt(filename_save,val_list,fmt='%s')
        filename_save=str(save_dir)+'pred_segs/'+'mean_dsc.txt'
        np.savetxt(filename_save,val_list_mean,fmt='%s')
        #filename_save=str(save_dir)+'pred_segs/'+'net_dsc_mean.txt'
        #net_mean_dsc=[]
        #net_mean_dsc.append(round(np.mean(val_list_mean),3))
        #np.savetxt(filename_save,net_mean_dsc,fmt='%s')

        if(print_assd_hd_scores==1):
            #for ASSD
            val_list=[]
            val_list_mean=[]
            #for HD
            hd_val_list=[]
            hd_val_list_mean=[]

            for i in range(0,self.num_classes-1):
                assd=assd_all[:,i]
                hd=hd_all[:,i]
                #ASSD
                #val_list.append(round(np.mean(assd), 3))
                val_list.append(round(np.median(assd), 3))
                val_list.append(round(np.std(assd), 3))
                val_list_mean.append(round(np.mean(assd), 3))
                filename_save=str(save_dir)+'pred_segs/'+str(struct_name[i])+'_20subjs_assd.txt'
                np.savetxt(filename_save,assd,fmt='%s')
                #HD
                #hd_val_list.append(round(np.mean(hd), 3))
                hd_val_list.append(round(np.median(hd), 3))
                hd_val_list.append(round(np.std(hd), 3))
                hd_val_list_mean.append(round(np.mean(hd), 3))
                filename_save=str(save_dir)+'pred_segs/'+str(struct_name[i])+'_20subjs_hd.txt'
                np.savetxt(filename_save,hd,fmt='%s')

            filename_save=str(save_dir)+'pred_segs/'+'median_std_assd.txt'
            np.savetxt(filename_save,val_list,fmt='%s')
            filename_save=str(save_dir)+'pred_segs/'+'assd_mean.txt'
            np.savetxt(filename_save,val_list_mean,fmt='%s')

            filename_save=str(save_dir)+'pred_segs/'+'median_std_hd.txt'
            np.savetxt(filename_save,hd_val_list,fmt='%s')
            filename_save=str(save_dir)+'pred_segs/'+'hd_mean.txt'
            np.savetxt(filename_save,hd_val_list_mean,fmt='%s')

    def plot_predicted_seg_ss(self, test_data_img,test_data_labels,predicted_labels,save_dir,test_id):
        '''
        To plot the original image, ground truth mask and predicted mask
        input params:
            test_data_img: test image to be plotted
            test_data_labels: test image GT mask to be plotted
            predicted_labels: predicted mask of the test image
            save_dir: directory where to save the plot
            test_id: patient id number of the dataset
        returns:
            None
        '''
        n_examples=3
        fig, axs = plt.subplots(3, n_examples, figsize=(10, 10))
        fig.suptitle('Predicted Seg',fontsize=10)
        for example_i in range(n_examples):
            if(example_i==0):
                axs[0][0].set_title('test image')
                axs[1][0].set_title('ground truth mask')
                axs[2][0].set_title('predicted mask')

            axs[0][example_i].imshow(test_data_img[:,:,example_i*2],cmap='gray')
            axs[1][example_i].imshow(test_data_labels[:,:,example_i*2])
            axs[2][example_i].imshow(np.squeeze(predicted_labels[:,:,example_i*2]))
            axs[0][example_i].axis('off')
            axs[1][example_i].axis('off')
            axs[2][example_i].axis('off')

        savefile_name=str(save_dir)+'tst'+str(test_id)+'_predicted_segmentation_masks.png'
        fig.savefig(savefile_name)
        plt.close('all')

    def plot_deformed_imgs(self,ld_img_batch,y_geo_deformed,flow_vec,save_dir,index):
        '''
        To plot the different deformation fields generated from different z's sampled.
        These deformation fields are applied on a single image to illustrate different augmented images that can be generated from a single image.
        input params:
            ld_img_batch: input labeled image
            y_geo_deformed: deformed images (non-affine spatial transformation applied)
            flow_vec: deformation fields
        returns:
            None
        '''
        save_dir_tmp=str(save_dir)+'/plots/'
        pathlib.Path(save_dir_tmp).mkdir(parents=True, exist_ok=True)

        savefile_name_tmp=str(save_dir_tmp)+'deformed_imgs_for_different_z_sampled_for_'
        max_val=5
        step_update=1

        #def for quiver plot
        X, Y = np.meshgrid(np.arange(0, self.img_size_x, 1), np.arange(0, self.img_size_y, 1))
        #every 10th arrow to plot
        t=10

        plt.figure(figsize=(18,6))
        plt.suptitle('orig vs deformed imgs')

        for i in range(0,max_val,step_update):
            train_slice=np.squeeze(ld_img_batch[i,:,:,0])
            y_deformed_slice=np.squeeze(y_geo_deformed[i,:,:,0])
            v_x=np.squeeze(flow_vec[i,:,:,0])
            v_y=np.squeeze(flow_vec[i,:,:,1])

            if(i==0):
                plt.subplot(2, max_val+1, 1)
                plt.title('orig img')
                plt.imshow(train_slice,cmap='gray')
                plt.axis('off')

            plt.subplot(2, max_val+1, i+2)
            if(i==0):
                plt.title('deformation field over imgs -->')
            plt.imshow(train_slice,cmap='gray')
            plt.quiver(X[::t, ::t], Y[::t, ::t], v_x[::t, ::t], v_y[::t, ::t], pivot='mid', units='inches',color='yellow')
            plt.axis('off')

            plt.subplot(2, max_val+1, max_val+1+i+2)
            if(i==0):
                plt.title('deformed imgs -->')
            plt.imshow(y_deformed_slice,cmap='gray')
            plt.axis('off')

        savefile_name=str(savefile_name_tmp)+'i_'+str(index)+'.png'
        plt.savefig(savefile_name)
        plt.close('all')

    def plot_intensity_transformed_imgs(self,ld_img_batch,y_int_deformed,int_vec,save_dir,index):
        '''
        To plot the different intensity fields generated from different z's sampled.
        These intensity fields are applied on a single image to illustrate different augmented images that can be generated from a single image.
        input params:
            ld_img_batch: input labeled image
            y_int_deformed: intensity transformed images
            int_vec: intensity fields
        returns:
            None
        '''
        save_dir_tmp=str(save_dir)+'/plots/'
        pathlib.Path(save_dir_tmp).mkdir(parents=True, exist_ok=True)

        savefile_name_tmp=str(save_dir_tmp)+'intensity_transformed_imgs_for_different_z_sampled_for_'
        max_val=5
        step_update=1

        plt.figure(figsize=(18,6))
        plt.suptitle('orig vs intensity transformed imgs')

        for i in range(0,max_val,step_update):
            train_slice=np.squeeze(ld_img_batch[i,:,:,0])
            y_deformed_slice=np.squeeze(y_int_deformed[i,:,:,0])
            int_slice=np.squeeze(int_vec[i,:,:,0])

            if(i==0):
                plt.subplot(2, max_val+1, 1)
                plt.title('orig img')
                plt.imshow(train_slice,cmap='gray')
                plt.axis('off')

            plt.subplot(2, max_val+1, i+2)
            if(i==0):
                plt.title('intensity fields -->')
            plt.imshow(int_slice,cmap='gray')
            plt.axis('off')

            plt.subplot(2, max_val+1, max_val+1+i+2)
            if(i==0):
                plt.title('intensity transformed imgs -->')
            plt.imshow(y_deformed_slice,cmap='gray')
            plt.axis('off')

        savefile_name=str(savefile_name_tmp)+'i_'+str(index)+'.png'
        plt.savefig(savefile_name)
        plt.close('all')

    def write_gif_func(self, ip_img, imsize, save_dir,index=0):
        '''
        To save a gif of the input stack of 2D slices
        input params:
            ip_img: input stack of 2D slices
            imsize: image dimensions
            save_dir:directory to save the gif
        returns:
            None
        '''
        y = np.squeeze(ip_img)
        y_t=np.transpose(y)
        recons_ims = np.reshape(y_t,(self.img_size_x*self.img_size_y,self.batch_size))

        dataset =np.transpose(recons_ims.reshape(1,imsize[0],imsize[1],recons_ims.shape[1]),[3,0,1,2])
        np.expand_dims(dataset, axis=1)
        dataset = np.tile(dataset, [1,3,1,1])
        imname=save_dir+'plots/test_slice_index_'+str(index)+'.gif'
        write_gif((dataset*256).astype(np.uint8), imname, fps=5)

