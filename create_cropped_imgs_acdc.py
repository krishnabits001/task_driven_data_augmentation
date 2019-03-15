'''
This script is to preprocess all the data into the chosen resolution and fixed dimensions specified in the init_acdc.py config file.
This re-sampled (to target resolution) and cropped/zero-padded image,label pairs are stored in npy files and are later used directly while the training of the network.
'''
import numpy as np
import pathlib

import argparse
parser = argparse.ArgumentParser()

#data set type
parser.add_argument('--dataset', type=str, default='acdc', choices=['acdc'])

parse_config = parser.parse_args()

if parse_config.dataset == 'acdc':
    #print('load acdc configs')
    import experiment_init.init_acdc as cfg
    import experiment_init.data_cfg_acdc as data_list
else:
    raise ValueError(parse_config.dataset)

from dataloaders import dataloaderObj
dt = dataloaderObj(cfg)

if parse_config.dataset == 'acdc':
    #print('set acdc orig img dataloader handle')
    orig_img_dt=dt.load_acdc_imgs

#loop over all image,label pairs to create cropped image,label pairs
for index in range(1,100):
    # Load each image
    if(index<10):
        test_id='00'+str(index)
    elif(index<100):
        test_id='0'+str(index)
    else:
        test_id=str(index)
    test_id_l=[test_id]

    #load image,label pairs and process it to chosen resolution and dimensions    
    img_sys,label_sys,pixel_size,affine_tst= orig_img_dt(test_id_l,ret_affine=1)
    cropped_img_sys,cropped_mask_sys = dt.preprocess_data(img_sys, label_sys, pixel_size)

    #save the processed cropped img and its label
    save_dir_tmp=str(cfg.data_path_tr_cropped)+'/'+str(test_id)+'/'
    pathlib.Path(save_dir_tmp).mkdir(parents=True, exist_ok=True)
    savefile_name=str(save_dir_tmp)+'img_cropped.npy' 
    np.save(savefile_name,cropped_img_sys)
    savefile_name=str(save_dir_tmp)+'mask_cropped.npy' 
    np.save(savefile_name,cropped_mask_sys)

