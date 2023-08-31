import numpy as np
import torch
from tqdm import tqdm

from skimage import metrics as skimage_metrics
from skimage.util import img_as_ubyte

# LPIPS metrics with AlexNet and VGG
import lpips
lpips_alex = lpips.LPIPS(net="alex", version="0.1")
lpips_vgg = lpips.LPIPS(net="vgg", version="0.1")

# Nanopyx metrics: Error map (RSE and RSP) and decorrelation analysis 
from nanopyx.core.transform.new_error_map import ErrorMap
from nanopyx.core.analysis.decorr import DecorrAnalysis

# ILNIQE (in a local file)
from .ILNIQE import calculate_ilniqe

def obtain_metrics(gt_image_list, predicted_image_list, wf_image_list, test_metric_indexes):
    """
    Calculate various metrics for evaluating the performance of an image prediction model.

    Args:
        gt_image_list (List[np.ndarray]): A list of ground truth images.
        predicted_image_list (List[np.ndarray]): A list of predicted images.
        wf_image_list (List[np.ndarray]): A list of wavefront images.
        test_metric_indexes (List[int]): A list of indexes to calculate additional metrics.

    Returns:
        dict: A dictionary containing different metrics as keys and their corresponding values as lists.

    Raises:
        AssertionError: If the minimum value of the wavefront image is greater than 0 or the maximum value is less than 0.

    Note:
        This function uses various image metrics including MSE, SSIM, PSNR, GT RSE, GT RSP, Pred RSE, Pred RSP, and Decorrelation.
        It also calculates metrics using the LPIPS (Learned Perceptual Image Patch Similarity) model, ILNIQE (Image Lab Non-Reference Image Quality Evaluation), and other metrics.
        The calculated metrics are stored in a dictionary with the metric names as keys and lists of values as their corresponding values.
    """
    metrics_dict = {
        "ssim": [],
        "psnr": [],
        "mse": [],
        "alex": [],
        "vgg": [],
        "ilniqe": [],
        "fsim": [],
        "gmsd": [],
        "vsi": [],
        "haarpsi": [],
        "mdsi": [],
        "pieapp": [],
        "dists": [],
        "brisqe": [],
        "fid": [],
        "gt_rse":[],
        "gt_rsp":[],
        "pred_rse":[],
        "pred_rsp":[],
        "decor":[]
    }

    test_data_length = len(gt_image_list)

    for i in tqdm(range(test_data_length)):
        
        # Load the widefield image, ground truth image, and predicted image
        gt_image = gt_image_list[i][:, :, 0]
        predicted_image = predicted_image_list[i][:, :, 0]
        wf_image = wf_image_list[i][:, :, 0]


        # Print info about the images
        print(
            f"gt_image: {gt_image.shape} - {gt_image.min()} {gt_image.max()} - {gt_image.dtype}"
        )
        print(
            f"predicted_image: {predicted_image.shape} - {predicted_image.min()} {predicted_image.max()} - {predicted_image.dtype}"
        )
        print(
            f"wf_image: {wf_image.shape} - {wf_image.min()} {wf_image.max()} - {wf_image.dtype}"
        )


        # Convert the Numpy images into Pytorch tensors
        # Pass the images into Pytorch format (1, 1, X, X)
        gt_image_piq = np.expand_dims(gt_image, axis=0)
        gt_image_piq = np.expand_dims(gt_image_piq, axis=0)
        
        predicted_image_piq = np.expand_dims(predicted_image, axis=0)
        predicted_image_piq = np.expand_dims(predicted_image_piq, axis=0)

        # Pytorch does not support uint16
        if gt_image_piq.dtype == np.uint16:
            gt_image_piq = gt_image_piq.astype(np.uint8)
        if predicted_image_piq.dtype == np.uint16:
            predicted_image_piq = predicted_image_piq.astype(np.uint8) 
            
        # Convert the images into Pytorch tensors
        gt_image_piq = torch.from_numpy(gt_image_piq)
        predicted_image_piq = torch.from_numpy(predicted_image_piq)

        
        # Assert that there are no negative values
        assert wf_image.min() <= 0. and wf_image.max() >= 0.

        # In case all the predicted values are equal (all zeros for example)
        all_equals = np.all(predicted_image==np.ravel(predicted_image)[0])

    
        #####################################
        #
        # Calculate the skimage metrics

        metrics_dict["mse"].append(
            skimage_metrics.mean_squared_error(gt_image, predicted_image)
        )

        metrics_dict["ssim"].append(
            skimage_metrics.structural_similarity(
                predicted_image, gt_image, data_range=1.0
            )
        )
        metrics_dict["psnr"].append(
            skimage_metrics.peak_signal_noise_ratio(gt_image, predicted_image)
        )

        #
        #####################################

        #####################################
        #
        # Calculate the LPIPS metrics

        metrics_dict["alex"].append(
                np.squeeze(
                    lpips_alex(gt_image_piq.float(), predicted_image_piq.float())
                    .detach()
                    .numpy()
                )
            )
        metrics_dict["vgg"].append(
            np.squeeze(
                lpips_vgg(gt_image_piq.float(), predicted_image_piq.float())
                .detach()
                .numpy()
            )
        )

        #
        #####################################

        #####################################
        #
        # Calculate the Nanopyx metrics

        error_map = ErrorMap()
        error_map.optimise(wf_image, gt_image)
        metrics_dict["gt_rse"].append(
            error_map.getRSE()
        )
        metrics_dict["gt_rsp"].append(
            error_map.getRSP()
        )

        if not all_equals:
            error_map = ErrorMap()
            error_map.optimise(wf_image, predicted_image)
            metrics_dict["pred_rse"].append(
                error_map.getRSE()
            )
            metrics_dict["pred_rsp"].append(
                error_map.getRSP()
            )
        else: 
            metrics_dict["pred_rse"].append(np.nan)
            metrics_dict["pred_rsp"].append(np.nan)

        if not all_equals:
            decorr_calculator_raw = DecorrAnalysis()
            decorr_calculator_raw.run_analysis(predicted_image)
            metrics_dict["decor"].append(
                decorr_calculator_raw.resolution
            )
        else: 
            metrics_dict["decor"].append(np.nan)

        #
        #####################################

        #####################################
        #
        # Calculate the ILNIQE
        
        # Temporally commented to avoid long evaluation times (83 seconds for each image)
        # if not all_equals:
        #     metrics_dict['ilniqe'].append(calculate_ilniqe(img_as_ubyte(predicted_image), 0,
        #                                     input_order='HW', resize=True, version='python'))
        # else: 
        #     metrics_dict['ilniqe'].append(np.nan)

        #
        #####################################

        
        if i in test_metric_indexes:
            # In case you want to calculate in specific images (a reduced number to avoid time issues)
            pass
        

    return metrics_dict
