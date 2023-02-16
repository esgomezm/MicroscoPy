import numpy as np
import torch
from tqdm import tqdm
import math
import cv2
import os

from skimage import metrics as skimge_metrics
from skimage import img_as_ubyte

from ILNIQE import calculate_ilniqe

import lpips

lpips_alex = lpips.LPIPS(net='alex',version='0.1')
lpips_vgg = lpips.LPIPS(net='vgg',version='0.1')

import piq

dists_loss = piq.DISTS()
pieapp_loss = piq.PieAPP()

def obtain_metrics(gt_image_list, predicted_image_list, test_metric_indexes):
    metrics_dict = {'ssim':[], 'psnr':[], 'mse':[], 'alex':[], 
                    'vgg':[], 'ilniqe':[], 'fsim':[], 'gmsd':[], 
                    'vsi':[], 'haarpsi':[], 'mdsi':[], 'pieapp':[], 
                    'dists':[], 'brisqe':[]}
    
    test_data_length = len(gt_image_list)
    
    for i in tqdm(range(test_data_length)):
        
        gt_image = gt_image_list[i,:,:,0]
        predicted_image = predicted_image_list[i,:,:,0]
            
        gt_image_piq = np.expand_dims(gt_image, axis=0)
        gt_image_piq = np.expand_dims(gt_image_piq, axis=0)
        gt_image_piq = torch.from_numpy(gt_image_piq)
        
        predicted_image_piq = np.expand_dims(predicted_image, axis=0)
        predicted_image_piq = np.expand_dims(predicted_image_piq, axis=0)
        predicted_image_piq = torch.from_numpy(predicted_image_piq)
        
        metrics_dict['ssim'].append(skimge_metrics.structural_similarity(predicted_image, gt_image))
        metrics_dict['psnr'].append(skimge_metrics.peak_signal_noise_ratio(gt_image, predicted_image))
        metrics_dict['mse'].append(skimge_metrics.mean_squared_error(gt_image, predicted_image))
        #metrics_dict['fsim'].append(piq.fsim(predicted_image_piq, gt_image_piq, chromatic=False).item())
        #metrics_dict['gmsd'].append(piq.gmsd(predicted_image_piq, gt_image_piq).item())
        #metrics_dict['vsi'].append(piq.vsi(predicted_image_piq, gt_image_piq).item())
        #metrics_dict['haarpsi'].append(piq.haarpsi(predicted_image_piq, gt_image_piq).item())
        #metrics_dict['mdsi'].append(piq.mdsi(predicted_image_piq, gt_image_piq).item())
        #metrics_dict['dists'].append(dists_loss(predicted_image_piq, gt_image_piq).item())
        #metrics_dict['brisqe'].append(piq.brisque(predicted_image_piq).item())
        if i in test_metric_indexes:
            metrics_dict['alex'].append(np.squeeze(lpips_alex(gt_image_piq.float(), predicted_image_piq.float()).detach().numpy()))
            metrics_dict['vgg'].append(np.squeeze(lpips_vgg(gt_image_piq.float(), predicted_image_piq.float()).detach().numpy()))
            #metrics_dict['ilniqe'].append(calculate_ilniqe(img_as_ubyte(predicted_image), 0, 
            #                                  input_order='HW', resize=True, version='python'))
            #metrics_dict['pieapp'].append(pieapp_loss(predicted_image_piq.float(), gt_image_piq.float()).item())
            
    return metrics_dict
