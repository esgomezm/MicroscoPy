import numpy as np
import torch
from tqdm import tqdm

from skimage import metrics as skimage_metrics
from skimage.util import img_as_ubyte
#from torchmetrics.image.fid import FrechetInceptionDistance

#from .ILNIQE import calculate_ilniqe

import lpips

lpips_alex = lpips.LPIPS(net='alex',version='0.1')
lpips_vgg = lpips.LPIPS(net='vgg',version='0.1')

# import piq

# dists_loss = piq.DISTS()
# pieapp_loss = piq.PieAPP()

def obtain_metrics(gt_image_list, predicted_image_list, test_metric_indexes):
    metrics_dict = {'ssim':[], 'psnr':[], 'mse':[], 'alex':[], 
                    'vgg':[], 'ilniqe':[], 'fsim':[], 'gmsd':[], 
                    'vsi':[], 'haarpsi':[], 'mdsi':[], 'pieapp':[], 
                    'dists':[], 'brisqe':[], 'fid':[]}
    
    test_data_length = len(gt_image_list)
    
    for i in tqdm(range(test_data_length)):
        
        gt_image = gt_image_list[i][:,:,0]
        predicted_image = predicted_image_list[i][:,:,0]

        print('gt_image.shape: {}'.format(gt_image.shape))
        print('predicted_image.shape: {}'.format(predicted_image.shape))
            
        gt_image_piq = np.expand_dims(gt_image, axis=0)
        gt_image_piq = np.expand_dims(gt_image_piq, axis=0)
        if gt_image_piq.dtype == np.uint16:
            gt_image_piq = gt_image_piq.astype(np.uint8) # Pytorch does not support uint16
        gt_image_piq = torch.from_numpy(gt_image_piq)
        
        predicted_image_piq = np.expand_dims(predicted_image, axis=0)
        predicted_image_piq = np.expand_dims(predicted_image_piq, axis=0)
        if predicted_image_piq.dtype == np.uint16:
            predicted_image_piq = predicted_image_piq.astype(np.uint8) # Pytorch does not support uint16
        predicted_image_piq = torch.from_numpy(predicted_image_piq)

        '''
        gt_image_piq_3c = np.expand_dims(gt_image, axis=0)
        gt_image_piq_3c = np.concatenate((gt_image_piq_3c,gt_image_piq_3c,gt_image_piq_3c), axis=0)
        gt_image_piq_3c = np.expand_dims(gt_image_piq_3c, axis=0)
        gt_image_piq_3c = torch.from_numpy(gt_image_piq_3c)

        predicted_image_piq_3c = np.expand_dims(predicted_image, axis=0)
        predicted_image_piq_3c = np.concatenate((predicted_image_piq_3c,predicted_image_piq_3c,predicted_image_piq_3c), axis=0)
        predicted_image_piq_3c = np.expand_dims(predicted_image_piq_3c, axis=0)
        predicted_image_piq_3c = torch.from_numpy(predicted_image_piq_3c)
        '''

        # the input is expected to be mini-batches of 3-channel RGB images of shape (3 x H x W)
        # All images will be resized to 299 x 299 which is the size of the original training data.
        #fid = FrechetInceptionDistance(feature=64, normalize=True) # feature=64,192,768,2048 normalize=False(uint8),True(float)
        #fid.update(gt_image_piq_3c, real=True)
        #fid.update(predicted_image_piq_3c, real=False)
        #fid.compute()
       
        print(f'gt_image: {gt_image.shape} - {gt_image.min()} {gt_image.max()} - {gt_image.dtype}')
        print(f'predicted_image: {predicted_image.shape} - {predicted_image.min()} {predicted_image.max()} - {predicted_image.dtype}')


        metrics_dict['mse'].append(skimage_metrics.mean_squared_error(gt_image, predicted_image))
        metrics_dict['ssim'].append(skimage_metrics.structural_similarity(predicted_image, gt_image, data_range=1.0))
        metrics_dict['psnr'].append(skimage_metrics.peak_signal_noise_ratio(gt_image, predicted_image))
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
