import numpy as np
from skimage import filters
from skimage import transform
from skimage.util import random_noise
from scipy.ndimage.interpolation import zoom as npzoom
from skimage.transform import rescale

from .utils import min_max_normalization as norm

def add_poisson_noise(img, lam=1.0):
    """
    Generates a noisy image by adding Poisson noise to the input image.

    Parameters:
        img (ndarray): The input image.
        lam (float): The parameter lambda for the Poisson distribution. Default is 1.0.

    Returns:
        ndarray: The noisy image with Poisson noise added.
    """
    poisson_noise = np.random.poisson(lam=lam, size=img.shape)
    noisy_img = img + norm(poisson_noise)
    return norm(noisy_img)

#####################################
#
# Create corresponding training patches synthetically by adding noise
# and downsampling the images (see https://www.biorxiv.org/content/10.1101/740548v3)
# Based on the code on: https://github.com/BPHO-Salk/PSSR/blob/master/utils/crappifiers.py

def downsampleonly(x, scale=4):
    return npzoom(x, 1 / scale, order=1)

def fluo_G_D(x, scale=4):
    x = npzoom(x, 1 / scale, order=1)

    mu, sigma = 0, 5
    noise = np.random.normal(mu, sigma * 0.05, x.shape)
    x = np.clip(x + noise, 0, 1)

    return x

def fluo_AG_D(x, scale=4):
    x = npzoom(x, 1 / scale, order=1)

    lvar = filters.gaussian(x, sigma=5) + 1e-10
    x = random_noise(x, mode="localvar", local_vars=lvar * 0.5)

    return x

def fluo_SP_D(x, scale=4):
    x = npzoom(x, 1 / scale, order=1)

    x = random_noise(x, mode="salt", amount=0.005)
    x = random_noise(x, mode="pepper", amount=0.005)

    return x

def fluo_SP_AG_D_sameas_preprint(x, scale=4):
    x = npzoom(x, 1 / scale, order=1)

    x = random_noise(x, mode="salt", amount=0.005)
    x = random_noise(x, mode="pepper", amount=0.005)
    lvar = filters.gaussian(x, sigma=5) + 1e-10
    x = random_noise(x, mode="localvar", local_vars=lvar * 0.5)

    return x

def fluo_SP_AG_D_sameas_preprint_rescale(x, scale=4):
    x = rescale(x, scale=1 / scale, order=1)

    x = random_noise(x, mode="salt", amount=0.005)
    x = random_noise(x, mode="pepper", amount=0.005)
    lvar = filters.gaussian(x, sigma=5) + 1e-10
    x = random_noise(x, mode="localvar", local_vars=lvar * 0.5)

    return x

def em_AG_D_sameas_preprint(x, scale=4):
    x = npzoom(x, 1 / scale, order=1)

    lvar = filters.gaussian(x, sigma=3) + 1e-10
    x = random_noise(x, mode="localvar", local_vars=lvar * 0.05)

    return x

def em_G_D_001(x, scale=4):
    x = npzoom(x, 1 / scale, order=1)

    noise = np.random.normal(0, 3, x.shape)
    x = x + noise
    x = x - x.min()
    x = x / x.max()

    return x

def em_G_D_002(x, scale=4):
    x = npzoom(x, 1 / scale, order=1)

    mu, sigma = 0, 3
    noise = np.random.normal(mu, sigma * 0.05, x.shape)
    x = np.clip(x + noise, 0, 1)

    return x

def em_P_D_001(x, scale=4):
    x = npzoom(x, 1 / scale, order=1)
    x = random_noise(x, mode="poisson", seed=1)

    return x

def new_crap_AG_SP(x, scale=4):
    x = rescale(x, scale=1 / scale, order=1)

    lvar = filters.gaussian(x, sigma=5) + 1e-10
    x = norm(x)
    x = random_noise(x, mode="localvar", local_vars=lvar * 0.5)

    x = random_noise(x, mode="salt", amount=0.005)
    x = random_noise(x, mode="pepper", amount=0.005)

    return x

def new_crap(x, scale=4):
    x = rescale(x, scale=1 / scale, order=1)

    x = random_noise(x, mode="salt", amount=0.005)
    x = random_noise(x, mode="pepper", amount=0.005)
    lvar = filters.gaussian(x, sigma=5) + 1e-10
    x = random_noise(x, mode="localvar", local_vars=lvar * 0.5)

    return x

#
#####################################

#####################################
#
# New crappifiers created by me.

def em_old_crappify(img, scale):
  img = filters.gaussian(img, sigma=3) + 1e-10
  img = norm(img)
  return transform.resize(img, (img.shape[0]//scale, img.shape[1]//scale), order=1)

def em_poisson_crappify(img, scale, lam=1.0):
    img = transform.resize(img, (img.shape[0] // scale, img.shape[1] // scale), order=1)

    img = filters.gaussian(img, sigma=2) + 1e-10
    img = norm(img)
    img = add_poisson_noise(img, lam=lam)
    return img


def fluo_poisson_crappify(img, scale, lam=1.0):
    img = transform.resize(img, (img.shape[0] // scale, img.shape[1] // scale), order=1)

    img = filters.gaussian(img, sigma=2) + 1e-10
    img = norm(img)
    img = random_noise(img, mode="salt", amount=0.005)
    img = random_noise(img, mode="pepper", amount=0.005)
    img = add_poisson_noise(img, lam=lam)
    return img


def em_gaussian_crappify(img, scale, mu=-0.1, sigma=0.15, scalar=0.9):
    img = norm(npzoom(img, 1/scale, order=1))
    noise = np.random.normal(mu, sigma, img.shape)
    img = norm(img + scalar*noise)
    return img

def em_gaussian_crappify_s05(img, scale, mu=-0.1, sigma=0.15, scalar=0.5):
    img = norm(npzoom(img, 1/scale, order=1))
    noise = np.random.normal(mu, sigma, img.shape)
    img = norm(img + scalar*noise)
    return img

def em_gaussian_crappify_s03(img, scale, mu=-0.1, sigma=0.15, scalar=0.3):
    img = norm(npzoom(img, 1/scale, order=1))
    noise = np.random.normal(mu, sigma, img.shape)
    img = norm(img + scalar*noise)
    return img

#
#####################################

#####
# Function and dictionary that control the crappifiers
#####

def apply_crappifier(x, scale, crappifier_name):
    """
    Apply the specified crappifier to the input data.

    Parameters:
        x (numpy.ndarray): The input data.
        scale (float): The scaling factor for the crappifier.
        crappifier_name (str): The name of the crappifier to apply.

    Returns:
        numpy.ndarray: The output data after applying the crappifier.

    Raises:
        ValueError: If the specified crappifier_name is not in the CRAPPIFIER_DICT.
    """
    if crappifier_name in CRAPPIFIER_DICT:
        return norm(CRAPPIFIER_DICT[crappifier_name](norm(x), scale).astype(np.float32))
    else:
        raise ValueError(
            "The selected `{}` crappifier_name is not in: {}".format(
                crappifier_name, CRAPPIFIER_DICT.keys
            )
        )

# Dictionary with all the crappifier functions
CRAPPIFIER_DICT = {
    "downsampleonly": downsampleonly,
    "fluo_G_D": fluo_G_D,
    "fluo_AG_D": fluo_AG_D,
    "fluo_SP_D": fluo_SP_D,
    "fluo_SP_AG_D_sameas_preprint": fluo_SP_AG_D_sameas_preprint,
    "fluo_SP_AG_D_sameas_preprint_rescale": fluo_SP_AG_D_sameas_preprint_rescale,
    "em_AG_D_sameas_preprint": em_AG_D_sameas_preprint,
    "em_G_D_001": em_G_D_001,
    "em_G_D_002": em_G_D_002,
    "em_P_D_001": em_P_D_001,
    "new_crap_AG_SP": new_crap_AG_SP,
    "new_crap": new_crap,
    "em_poisson_crappify": em_poisson_crappify,
    "fluo_poisson_crappify": fluo_poisson_crappify,
    "em_gaussian_crappify": em_gaussian_crappify,
    "em_gaussian_crappify_s05": em_gaussian_crappify_s05,
    "em_gaussian_crappify_s03": em_gaussian_crappify_s03,
    "em_old_crappify": em_old_crappify
}