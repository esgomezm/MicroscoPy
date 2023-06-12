import pytest
import os

def test_set_seed():
    from microscopy.utils import set_seed
    set_seed(66)

    # TODO: check seeds in np, tf, torch, os

def test_yaml_functions():
    from microscopy.utils import save_yaml, load_yaml, update_yaml
    
    yaml_name = 'file.yaml'
    data = {'value1': 3, 'value2': 2}

    if os.path.exists(yaml_name):
        os.remove(yaml_name)
    
    assert not os.path.exists(yaml_name)

    save_yaml(data, yaml_name)
    assert os.path.exists(yaml_name)
    
    loaded_data = load_yaml(yaml_name)
    assert loaded_data['value1'] == 3
    assert loaded_data['value2'] == 2 

    loaded_data = update_yaml(yaml_name, 'value1', 33)
    loaded_data = load_yaml(yaml_name)
    assert loaded_data['value1'] == 33 
    assert loaded_data['value2'] == 2 

    if os.path.exists(yaml_name):
        os.remove(yaml_name)

def test_ssim_loss():
    from microscopy.utils import ssim_loss
    # TODO: find a way to test ssim_loss

def test_vgg_loss():
    from microscopy.utils import vgg_loss
    # TODO: find a way to test vgg_loss

def test_perceptual_loss():
    from microscopy.utils import perceptual_loss
    # TODO: find a way to test perceptual_loss

def test_get_emb():
    from microscopy.utils import get_emb
    # TODO:

def test_concatenate_encoding():
    from microscopy.utils import concatenate_encoding
    # TODO:

def test_calculate_pad_for_Unet():
    from microscopy.utils import calculate_pad_for_Unet

    padding = calculate_pad_for_Unet(lr_img_shape=(125,125,1), 
                                     depth_Unet=4, 
                                     is_pre=True, 
                                     scale=2)

def test_remove_padding_for_Unet():
    from microscopy.utils import remove_padding_for_Unet

    unpadded_img = None

def test_add_padding_for_Unet():
    from microscopy.utils import add_padding_for_Unet

