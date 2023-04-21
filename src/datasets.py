import numpy as np
import os
import torch
from torch.utils.data import Dataset
from skimage import io

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from skimage import transform

from . import crappifiers 

#####
# Functions to sample an image using probability density function
#####
# Functions from https://github.com/esgomezm/microscopy-dl-suite-tf/blob/fcb8870624208bfb72dc7aea18a90738a081217f/dl-suite/utils

def index_from_pdf(pdf_im):
    prob = np.copy(pdf_im)
    # Normalize values to create a pdf with sum = 1
    prob = prob.ravel() / np.sum(prob)
    # Convert into a 1D pdf
    choices = np.prod(pdf_im.shape)
    index = np.random.choice(choices, size=1, p=prob)
    # Recover 2D shape
    coordinates = np.unravel_index(index, shape=pdf_im.shape)
    # Extract index
    indexh = coordinates[0][0]
    indexw = coordinates[1][0]
    return indexh, indexw

def sampling_pdf(y, pdf, height, width):
    h, w = y.shape[0], y.shape[1]

    if pdf == 1:
        indexw = np.random.randint(np.floor(width // 2),
                                   w - np.floor(width // 2))
        indexh = np.random.randint(np.floor(height // 2),
                                   h - np.floor(height // 2))
    else:
        print('img: {}x{}, crop: {}x{}'.format(y.shape[0], y.shape[1], height, width))
        # crop to fix patch size
        # croped_y = y[int(np.floor(height // 2)):-int(np.floor(height // 2)),
        #              int(np.floor(width // 2)) :-int(np.floor(width // 2))]
        # indexh, indexw = index_from_pdf(croped_y)

        kernel = np.ones((height,width))
        pdf = np.fft.irfft2(np.fft.rfft2(y) * np.fft.rfft2(kernel, y.shape))
        pdf_cropped = pdf[min(kernel.shape[0], pdf.shape[0]-1):, 
                          min(kernel.shape[1], pdf.shape[1]-1):]
        
        indexh, indexw = index_from_pdf(pdf_cropped)
        indexw = indexw + int(np.floor(width // 2))
        indexh = indexh + int(np.floor(height // 2))

    return indexh, indexw

#####
#
#####

def normalization(data, desired_accuracy=np.float32):
    return (data - data.min()) / (data.max() - data.min()).astype(desired_accuracy)

def read_image(filename, desired_accuracy=np.float32):
    return normalization(io.imread(filename), desired_accuracy=desired_accuracy)

def obtain_scale_factor(hr_filename, lr_filename, scale_factor, crappifier_name):

    if scale_factor is None and lr_filename is None:
        raise ValueError('A scale factor has to be given.')
    
    hr_img = read_image(hr_filename)

    if lr_filename is None:
        # If no path to the LR images is given, they will be artificially generated
        lr_img = normalization(crappifiers.apply_crappifier(hr_img, scale_factor, crappifier_name))
    else:
        lr_img = read_image(lr_filename)

    images_scale_factor = hr_img.shape[0]//lr_img.shape[0]

    return images_scale_factor if scale_factor is None else scale_factor

def read_image_pairs(hr_filename, lr_filename, scale_factor, crappifier_name):
    
    hr_img = read_image(hr_filename)

    if lr_filename is None:
        # If no path to the LR images is given, they will be artificially generated
        lr_img = normalization(crappifiers.apply_crappifier(hr_img, scale_factor, crappifier_name))
    else:
        lr_img = read_image(lr_filename)

        images_scale_factor = hr_img.shape[0]//lr_img.shape[0]

        if scale_factor > images_scale_factor:
            lr_img = normalization(crappifiers.apply_crappifier(lr_img, scale_factor//images_scale_factor, crappifier_name))

    return hr_img, lr_img

def extract_random_patches_from_image(hr_filename, lr_filename, scale_factor, 
                                      crappifier_name, lr_patch_shape, num_patches,
                                      datagen_sampling_pdf):

    hr_img, lr_img = read_image_pairs(hr_filename, lr_filename, scale_factor, crappifier_name)

    if lr_patch_shape is None:
        lr_patch_size_width = lr_img.shape[0]
        lr_patch_size_height = lr_img.shape[1]
    else: 
        lr_patch_size_width = lr_patch_shape[0]
        lr_patch_size_height = lr_patch_shape[1]

    if lr_img.shape[0] < lr_patch_size_width or hr_img.shape[0] < lr_patch_size_width * scale_factor:
        raise ValueError('Patch size is bigger than the given images.')

    lr_patches = []
    hr_patches = []

    for _ in range(num_patches):
        
        lr_idx_width, lr_idx_height = sampling_pdf(y=lr_img, pdf=datagen_sampling_pdf, 
                                                   height=lr_patch_size_height, width=lr_patch_size_width)

        print('lr_idx_width: {}, lr_idx_height: {}'.format(lr_idx_width, lr_idx_height))
        lr = lr_idx_height - np.floor(lr_patch_size_height // 2)
        lr = lr.astype(np.int)
        ur = lr_idx_height + np.round(lr_patch_size_height // 2)
        ur = ur.astype(np.int)

        lc = lr_idx_width - np.floor(lr_patch_size_width // 2)
        lc = lc.astype(np.int)
        uc = lr_idx_width + np.round(lr_patch_size_width // 2)
        uc = uc.astype(np.int)
        
        print('lr: {}, ur: {}'.format(lr, ur))
        print('lc: {}, uc: {}'.format(lc, uc))
        
        lr_patches.append(lr_img[lr:ur, lc:uc])
        hr_patches.append(hr_img[lr*scale_factor:ur*scale_factor, 
                                 lc*scale_factor:uc*scale_factor])
        print('lr_patches[-1].shape: {}'.format(lr_patches[-1].shape))
        print('hr_patches[-1].shape: {}'.format(hr_patches[-1].shape))

    return np.array(lr_patches), np.array(hr_patches)

def extract_random_patches_from_folder(hr_data_path, lr_data_path, filenames, scale_factor, 
                                      crappifier_name, lr_patch_shape, num_patches, datagen_sampling_pdf):

    # First lets check what is the scale factor, in case None is given
    actual_scale_factor = obtain_scale_factor(hr_filename=os.path.join(hr_data_path, filenames[0]), 
                                              lr_filename=None if lr_data_path is None else os.path.join(lr_data_path, filenames[0]), 
                                              scale_factor=scale_factor, 
                                              crappifier_name=crappifier_name)

    final_lr_patches = []
    final_hr_patches = []
    
    for f in filenames:
        hr_image_path = os.path.join(hr_data_path, f)
        if lr_data_path is not None:
            lr_image_path = os.path.join(lr_data_path, f)
        else:
            lr_image_path = None
        lr_patches, hr_patches = extract_random_patches_from_image(hr_image_path, lr_image_path, actual_scale_factor, 
                                                                   crappifier_name, lr_patch_shape, num_patches, datagen_sampling_pdf)
        final_lr_patches.append(lr_patches)
        final_hr_patches.append(hr_patches)

    final_lr_patches = np.concatenate(final_lr_patches)
    final_hr_patches = np.concatenate(final_hr_patches)

    return final_lr_patches, final_hr_patches, actual_scale_factor

# Random rotation of an image by a multiple of 90 degrees
def random_90rotation( img ):
    return transform.rotate(img, 90*np.random.randint( 0, 5 ), preserve_range=True)

# Runtime data augmentation
def get_train_val_generators(X_data, Y_data, validation_split=0.25,
                             batch_size=32, seed=42, show_examples=False,
                             rotation=True, horizontal_flip=True, vertical_flip=True ):
    X_train, X_test, Y_train, Y_test = train_test_split(X_data,
                                                      Y_data,
                                                      train_size=1-validation_split,
                                                      test_size=validation_split,
                                                      random_state=seed, shuffle=False)

    random_rotation=random_90rotation
    if not rotation:
        random_rotation=None

    # Image data generator distortion options
    data_gen_args = dict( preprocessing_function=random_rotation,
                        horizontal_flip=horizontal_flip,
                        vertical_flip=vertical_flip,
                        fill_mode='reflect')

    # Train data, provide the same seed and keyword arguments to the fit and flow methods
    X_datagen = ImageDataGenerator(**data_gen_args)
    Y_datagen = ImageDataGenerator(**data_gen_args)
    X_datagen.fit(X_train, augment=True, seed=seed)
    Y_datagen.fit(Y_train, augment=True, seed=seed)
    X_train_augmented = X_datagen.flow(X_train, batch_size=batch_size, shuffle=True, seed=seed)
    Y_train_augmented = Y_datagen.flow(Y_train, batch_size=batch_size, shuffle=True, seed=seed)


    # Validation data, no data augmentation, but we create a generator anyway
    X_datagen_val = ImageDataGenerator()
    Y_datagen_val = ImageDataGenerator()
    X_datagen_val.fit(X_test, augment=True, seed=seed)
    Y_datagen_val.fit(Y_test, augment=True, seed=seed)
    X_test_augmented = X_datagen_val.flow(X_test, batch_size=batch_size, shuffle=False, seed=seed)
    Y_test_augmented = Y_datagen_val.flow(Y_test, batch_size=batch_size, shuffle=False, seed=seed)
  
    if show_examples:
        plt.figure(figsize=(10,10))
        # generate samples and plot
        for i in range(9):
            # define subplot
            plt.subplot(330 + 1 + i)
            # generate batch of images
            batch = X_train_augmented.next()
            # convert to unsigned integers for viewing
            image = batch[0]
            # plot raw pixel data
            plt.imshow(image[:,:,0], vmin=0, vmax=1, cmap='gray')
        # show the figure
        plt.show()
        X_train_augmented.reset()
  
    # combine generators into one which yields image and masks
    train_generator = zip(X_train_augmented, Y_train_augmented)
    test_generator = zip(X_test_augmented, Y_test_augmented)

    return train_generator, test_generator

def get_generator(X_data, Y_data, batch_size=32, seed=42, 
                  show_examples=False,rotation=True, 
                  horizontal_flip=True, vertical_flip=True):

    random_rotation=random_90rotation
    if not rotation:
        random_rotation=None

    # Image data generator distortion options
    data_gen_args = dict( preprocessing_function=random_rotation,
                        horizontal_flip=horizontal_flip,
                        vertical_flip=vertical_flip,
                        fill_mode='reflect')

    # Train data, provide the same seed and keyword arguments to the fit and flow methods
    X_datagen = ImageDataGenerator(**data_gen_args)
    Y_datagen = ImageDataGenerator(**data_gen_args)
    X_datagen.fit(X_data, augment=True, seed=seed)
    Y_datagen.fit(Y_data, augment=True, seed=seed)
    X_train_augmented = X_datagen.flow(X_data, batch_size=batch_size, shuffle=True, seed=seed)
    Y_train_augmented = Y_datagen.flow(Y_data, batch_size=batch_size, shuffle=True, seed=seed)

    if show_examples:
        plt.figure(figsize=(10,10))
        # generate samples and plot
        for i in range(9):
            # define subplot
            plt.subplot(330 + 1 + i)
            # generate batch of images
            batch = X_train_augmented.next()
            # convert to unsigned integers for viewing
            image = batch[0]
            # plot raw pixel data
            plt.imshow(image[:,:,0], vmin=0, vmax=1, cmap='gray')
        # show the figure
        plt.show()
        X_train_augmented.reset()
  
    # combine generators into one which yields image and masks
    train_generator = zip(X_train_augmented, Y_train_augmented)

    return train_generator

#### Pytorch dataset ####

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        hr, lr = sample['hr'], sample['lr']

        # Pytorch is (batch, channels, width, height)
        hr = hr.transpose((2, 0, 1))
        lr = lr.transpose((2, 0, 1))
        return {'hr': torch.from_numpy(hr),
                'lr': torch.from_numpy(lr)}

class RandomHorizontalFlip(object):
    """Random horizontal flip"""

    def __init__(self):
        self.rng = np.random.default_rng()

    def __call__(self, sample):
        hr, lr = sample['hr'], sample['lr']

        if self.rng.random() < 0.5:
            hr = np.flip(hr, 1)
            lr = np.flip(lr, 1)

        return {'hr': hr.copy(),
                'lr': lr.copy()}

class RandomVerticalFlip(object):
    """Random vertical flip"""

    def __init__(self):
        self.rng = np.random.default_rng()

    def __call__(self, sample):
        hr, lr = sample['hr'], sample['lr']

        if self.rng.random() < 0.5:
            hr = np.flip(hr, 0)
            lr = np.flip(lr, 0)

        return {'hr': hr.copy(),
                'lr': lr.copy()}

class RandomRotate(object):
    """Random rotation"""

    def __init__(self):
        self.rng = np.random.default_rng()

    def __call__(self, sample):
        hr, lr = sample['hr'], sample['lr']

        k = self.rng.integers(4)

        hr = np.rot90(hr, k=k)
        lr = np.rot90(lr, k=k)

        return {'hr': hr.copy(),
                'lr': lr.copy()}

class PytorchDataset(Dataset):
    ''' Pytorch's Dataset type object used to obtain the train and 
        validation information during the training process. Saves the 
        filenames as an attribute and only loads the ones rquired for
        the training batch, reducing the required RAM memory during 
        and after the training.
    '''
    def __init__(self, hr_data_path, lr_data_path, filenames, 
                 scale_factor, crappifier_name, lr_patch_shape, 
                 num_patches, transformations, datagen_sampling_pdf,
                 val_split=None, validation=False):

        self.hr_data_path = hr_data_path
        self.lr_data_path = lr_data_path

        if val_split is None:
            self.filenames = filenames
        elif validation:
            self.filenames = filenames[:int(val_split*len(filenames))]
        else:
            self.filenames = filenames[int(val_split*len(filenames)):]

        self.transformations = transformations
        self.scale_factor = scale_factor
        self.crappifier_name = crappifier_name
        self.lr_patch_shape = lr_patch_shape

        self.num_patches = num_patches
        self.datagen_sampling_pdf = datagen_sampling_pdf

    def __len__(self):
        return self.num_patches * len(self.filenames)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        filename_idx = idx // self.num_patches
        hr_filename = os.path.join(self.hr_data_path, self.filenames[filename_idx]) 
        lr_filename = None if self.lr_data_path is None else os.path.join(self.lr_data_path, self.filenames[filename_idx]) 

        lr_patch, hr_patch = extract_random_patches_from_image(hr_filename, lr_filename, 
                                    self.scale_factor, self.crappifier_name, 
                                    self.lr_patch_shape, 1, self.datagen_sampling_pdf)

        lr_patch = np.expand_dims(lr_patch[0], axis=-1)
        hr_patch = np.expand_dims(hr_patch[0], axis=-1)

        sample = {'hr': hr_patch, 'lr': lr_patch}

        if self.transformations:
            sample = self.transformations(sample)

        return sample
