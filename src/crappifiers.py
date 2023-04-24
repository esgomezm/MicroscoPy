import numpy as np
from skimage import filters
from skimage import transform
from skimage.util import random_noise
from scipy.ndimage.interpolation import zoom as npzoom
from skimage.transform import rescale

def norm(data):
  return (data - data.min()) / (data.max() - data.min())

def add_poisson_noise(img, lam=1.):
    poisson_noise = np.random.poisson(lam=lam, size=img.shape)
    noisy_img = img + norm(poisson_noise)
    return (noisy_img - noisy_img.min()) / (noisy_img.max() - noisy_img.min())

# Create corresponding training patches synthetically by adding noise
# and downsampling the images (see https://www.biorxiv.org/content/10.1101/740548v3)

def downsampleonly(x, scale=4):
    return npzoom(x, 1/scale, order=1)

def fluo_G_D(x, scale=4):
    x = npzoom(x, 1/scale, order=1)

    mu, sigma = 0, 5
    noise = np.random.normal(mu, sigma*0.05, x.shape)
    x = np.clip(x + noise, 0, 1)
    
    return x

def fluo_AG_D(x, scale=4):
    x = npzoom(x, 1/scale, order=1)

    lvar = filters.gaussian(x, sigma=5) + 1e-10
    x = random_noise(x, mode='localvar', local_vars=lvar*0.5)

    return x

def fluo_SP_D(x, scale=4):
    x = npzoom(x, 1/scale, order=1)

    x = random_noise(x, mode='salt', amount=0.005)
    x = random_noise(x, mode='pepper', amount=0.005)

    return x

def fluo_SP_AG_D_sameas_preprint(x, scale=4):
    x = npzoom(x, 1/scale, order=1)

    x = random_noise(x, mode='salt', amount=0.005)
    x = random_noise(x, mode='pepper', amount=0.005)
    lvar = filters.gaussian(x, sigma=5) + 1e-10
    x = random_noise(x, mode='localvar', local_vars=lvar*0.5)

    return x

def fluo_SP_AG_D_sameas_preprint_rescale(x, scale=4):
    x = rescale(x, scale=1/scale, order=1, multichannel=len(x.shape) > 2)

    x = random_noise(x, mode='salt', amount=0.005)
    x = random_noise(x, mode='pepper', amount=0.005)
    lvar = filters.gaussian(x, sigma=5) + 1e-10
    x = random_noise(x, mode='localvar', local_vars=lvar*0.5)

    return x

def em_AG_D_sameas_preprint(x, scale=4):
    x = npzoom(x, 1/scale, order=1)

    lvar = filters.gaussian(x, sigma=3) + 1e-10
    x = random_noise(x, mode='localvar', local_vars=lvar*0.05)
    
    return x

def em_G_D_001(x, scale=4):
    x = npzoom(x, 1/scale, order=1)

    noise = np.random.normal(0, 3, x.shape)
    x = x + noise
    x = x - x.min()
    x = x/x.max()
    
    return x

def em_G_D_002(x, scale=4):
    x = npzoom(x, 1/scale, order=1)

    mu, sigma = 0, 3
    noise = np.random.normal(mu, sigma*0.05, x.shape)
    x = np.clip(x + noise, 0, 1)
    
    return x

def em_P_D_001(x, scale=4):
    x = npzoom(x, 1/scale, order=1)
    x = random_noise(x, mode='poisson', seed=1)

    return x

def new_crap_AG_SP(x, scale=4):
    x = rescale(x, scale=1/scale, order=1, multichannel=len(x.shape) > 2)

    lvar = filters.gaussian(x, sigma=5) + 1e-10
    x = random_noise(x, mode='localvar', local_vars=lvar*0.5)

    x = random_noise(x, mode='salt', amount=0.005)
    x = random_noise(x, mode='pepper', amount=0.005)

    return x

def new_crap(x, scale=4):
    x = rescale(x, scale=1/scale, order=1, multichannel=len(x.shape) > 2)

    x = random_noise(x, mode='salt', amount=0.005)
    x = random_noise(x, mode='pepper', amount=0.005)
    lvar = filters.gaussian(x, sigma=5) + 1e-10
    x = random_noise(x, mode='localvar', local_vars=lvar*0.5)
        
    return x 

# Create corresponding training patches synthetically by adding noise
# and downsampling the images (see https://www.biorxiv.org/content/10.1101/740548v3)
def em_crappify(img, scale):
    img = transform.resize(img, (img.shape[0]//scale, img.shape[1]//scale), order=1)

    img = filters.gaussian(img, sigma=3) + 1e-10
    #return npzoom(img, 1/scale, order=1)
    return img

def fluo_crappify(img,scale):
    img = transform.resize(img, (img.shape[0]//scale, img.shape[1]//scale), order=1)

    img = random_noise(img, mode='salt', amount=0.005)
    img = random_noise(img, mode='pepper', amount=0.005)
    img = filters.gaussian(img, sigma=5) + 1e-10
    #return npzoom(img, 1/scale, order=1)
    return img

def em_poisson_crappify(img, scale, lam=1.):
    img = transform.resize(img, (img.shape[0]//scale, img.shape[1]//scale), order=1)

    img = filters.gaussian(img, sigma=3) + 1e-10
    img = add_poisson_noise(img, lam=lam)
    return img

def fluo_poisson_crappify(img,scale, lam=1.):
    img = transform.resize(img, (img.shape[0]//scale, img.shape[1]//scale), order=1)

    img = filters.gaussian(img, sigma=5) + 1e-10
    img = random_noise(img, mode='salt', amount=0.005)
    img = random_noise(img, mode='pepper', amount=0.005)
    img = add_poisson_noise(img, lam=lam)
    return img

def apply_crappifier(x, scale, crappifier_name):
    crappifier_dict = { 'downsampleonly':downsampleonly,
                        'fluo_G_D':fluo_G_D, 
                        'fluo_AG_D':fluo_AG_D,
                        'fluo_SP_D':fluo_SP_D,
                        'fluo_SP_AG_D_sameas_preprint':fluo_SP_AG_D_sameas_preprint,
                        'fluo_SP_AG_D_sameas_preprint_rescale':fluo_SP_AG_D_sameas_preprint_rescale,
                        'em_AG_D_sameas_preprint':em_AG_D_sameas_preprint,
                        'em_G_D_001':em_G_D_001,
                        'em_G_D_002':em_G_D_002,
                        'em_P_D_001':em_P_D_001,
                        'new_crap_AG_SP':new_crap_AG_SP,
                        'new_crap':new_crap,
                        'em_crappify':em_crappify,
                        'fluo_crappify':fluo_crappify, 
                        'em_poisson_crappify':em_poisson_crappify,
                        'fluo_poisson_crappify':fluo_poisson_crappify, 
                      }
    
    if crappifier_name in crappifier_dict:
        return crappifier_dict[crappifier_name](x, scale).astype(np.float32)
    else:
        raise ValueError('The selected `{}` crappifier_name is not in: {}'.format(crappifier_name, crappifier_dict.keys))


