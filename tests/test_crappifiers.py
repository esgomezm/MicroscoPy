# import pytest
import numpy as np

def test_norm():
    from microscopy.crappifiers import norm
    data = np.random.randint(0, 255, (255,255,3))
    norm_data = norm(data)

    assert norm_data.max() <= 1. and norm_data.max() > 0.999, f'Randint - Maximum value: {norm_data.max()}'
    assert norm_data.min() == 0., f'Randint - Minimum value: {norm_data.min()}'

    zeros_data = np.zeros((255,255,3))
    norm_zeros_data = norm(zeros_data)

    assert norm_zeros_data.max() == 0., f'Zeros - Maximum value: {norm_zeros_data.max()}'
    assert norm_zeros_data.min() == 0., f'Zeros - Minimum value: {norm_zeros_data.min()}'
    
def test_add_poisson_noise():
    from microscopy.crappifiers import add_poisson_noise
    zeros_data = np.zeros((255,255,3))
    noisy_data = add_poisson_noise(zeros_data)

    assert noisy_data.max() > 0.

def test_crappifiers_dict():
    from microscopy.crappifiers import CRAPPIFIER_DICT
    amount_of_crappifiers = 16

    error_msg = "The ammount of crappify functions has changed"
    if len(CRAPPIFIER_DICT) < amount_of_crappifiers:
        error_msg = ", you need to remove old tests."
    if len(CRAPPIFIER_DICT) > amount_of_crappifiers:
        error_msg = ", you need to add new tests."

    assert len(CRAPPIFIER_DICT) == amount_of_crappifiers, error_msg

def test_crappifiers():
    from microscopy.crappifiers import CRAPPIFIER_DICT, apply_crappifier

    data = np.random.randint(0, 255, (255,255))

    for scale in [1,2,4,8]:
        for crappifier_name in CRAPPIFIER_DICT.keys():
            crappified_data = apply_crappifier(data, scale, crappifier_name)

            desired_shape = [x // scale + x % scale for x in data.shape] 

            #assert data.shape[-1] == crappified_data.shape[-1]
            assert desired_shape[0] == crappified_data.shape[0], f'{crappifier_name} - Desired shape: {desired_shape[0]} - Crappified shape: {crappified_data.shape[0]}'
            assert desired_shape[1] == crappified_data.shape[1], f'{crappifier_name} - Desired shape: {desired_shape[1]} - Crappified shape: {crappified_data.shape[1]}'

            assert crappified_data.max() <= 1. and crappified_data.max() > 0.999, f'{crappifier_name} - Maximum value: {crappified_data.max()}'
            assert crappified_data.min() == 0., f'{crappifier_name} - Minimum value: {crappified_data.min()}'