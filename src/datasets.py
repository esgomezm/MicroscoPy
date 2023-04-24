import numpy as np
import os
import torch
from torch.utils.data import Dataset
from skimage import io
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator

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
                                   max(w - np.floor(width // 2), np.floor(width // 2) + 1))
        indexh = np.random.randint(np.floor(height // 2),
                                   max(h - np.floor(height // 2), np.floor(height // 2) + 1))
    else:
        # crop to fix patch size
        # croped_y = y[int(np.floor(height // 2)):-int(np.floor(height // 2)),
        #              int(np.floor(width // 2)) :-int(np.floor(width // 2))]
        # indexh, indexw = index_from_pdf(croped_y)

        kernel = np.ones((height,width))
        pdf = np.fft.irfft2(np.fft.rfft2(y) * np.fft.rfft2(kernel, y.shape))
        pdf = (pdf - pdf.min()) / (pdf.max() - pdf.min())
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

        lr = lr_idx_height - np.floor(lr_patch_size_height // 2)
        lr = lr.astype(np.int)
        ur = lr_idx_height + np.round(lr_patch_size_height // 2)
        ur = ur.astype(np.int)

        lc = lr_idx_width - np.floor(lr_patch_size_width // 2)
        lc = lc.astype(np.int)
        uc = lr_idx_width + np.round(lr_patch_size_width // 2)
        uc = uc.astype(np.int)
        
        lr_patches.append(lr_img[lc:uc, lr:ur])
        hr_patches.append(hr_img[lc*scale_factor:uc*scale_factor, 
                                 lr*scale_factor:ur*scale_factor])

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

#####
# TensorFlow dataset
#####

class DataGenerator(tf.keras.utils.Sequence):
    
    def __init__(self, filenames, hr_data_path, lr_data_path,
                 scale_factor, crappifier_name, lr_patch_shape,
                 num_patches, datagen_sampling_pdf, validation_split,
                 batch_size, rotation, horizontal_flip, vertical_flip, 
                 module='train', shuffle=True):
        """
        Suffle is used to take everytime a different
        sample from the list in a random way so the
        training order differs. We create two instances
        with the same arguments.
        """
        self.filenames = np.array(filenames)
        self.indexes = np.arange(len(self.filenames))

        self.hr_data_path = hr_data_path
        self.lr_data_path = lr_data_path
        self.scale_factor = scale_factor
        self.crappifier_name = crappifier_name
        self.lr_patch_shape = lr_patch_shape
        self.num_patches = num_patches
        self.datagen_sampling_pdf = datagen_sampling_pdf

        self.validation_split = validation_split
        self.batch_size = batch_size

        self.rotation = rotation
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip

        self.module = module
        self.shuffle = shuffle  # #
        self.on_epoch_end()

        self.actual_scale_factor = None

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.filenames) / self.batch_size))

    def get_sample(self, idx):
        x, y = self.__getitem__(idx)

        return x, y, self.actual_scale_factor

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.filenames))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        # Find list of IDs
        list_IDs_temp = [self.indexes[k] for k in indexes]
        # Generate data
        x, y = self.__data_generation(list_IDs_temp)
        return x, y

    def __preprocess(self, x, y):

        apply_rotation = (np.random.random() < 0.5) * self.rotation
        apply_horizontal_flip = (np.random.random() < 0.5) * self.horizontal_flip
        apply_vertical_flip = (np.random.random() < 0.5) * self.vertical_flip

        processed_x = np.copy(x)
        processed_y = np.copy(y)

        if apply_rotation:
            print('rotation')
            print(processed_y.shape)
            rotation_times = np.random.randint(0, 5)
            processed_x = np.rot90(processed_x, rotation_times)
            processed_y = np.rot90(processed_y, rotation_times)
            print(processed_y.shape)
        if apply_horizontal_flip:
            print('horizontal_flip')
            print(processed_y.shape)
            processed_x = np.flip(processed_x, axis=-2)
            processed_y = np.flip(processed_y, axis=-2)
            print(processed_y.shape)
        if apply_vertical_flip:
            print('vertical_flip')
            print(processed_y.shape)
            processed_x = np.flip(processed_x, axis=-3)
            processed_y = np.flip(processed_y, axis=-3)
            print(processed_y.shape)

        return processed_x, processed_y

    def __data_generation(self, list_IDs_temp):
        # 'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Generate data
        if self.module == 'train':
            aux_x, aux_y, actual_scalefactor = extract_random_patches_from_folder(self.hr_data_path, self.lr_data_path, 
                                                              self.filenames[list_IDs_temp], 
                                                              scale_factor=self.scale_factor, 
                                                              crappifier_name=self.crappifier_name, 
                                                              lr_patch_shape=self.lr_patch_shape, 
                                                              num_patches=self.num_patches, 
                                                              datagen_sampling_pdf=self.datagen_sampling_pdf)

            x = np.expand_dims(aux_x, axis=-1)
            y = np.expand_dims(aux_y, axis=-1)

            x, y = self.__preprocess(x, y)

            self.actual_scale_factor = actual_scalefactor

        elif self.module == 'test':
            # print("Creating validation data...")
            aux_x, aux_y, actual_scalefactor = extract_random_patches_from_folder(self.hr_data_path, self.lr_data_path, 
                                                      self.filenames[list_IDs_temp], 
                                                      scale_factor=self.scale_factor, 
                                                      crappifier_name=self.crappifier_name, 
                                                      lr_patch_shape=self.lr_patch_shape, 
                                                      num_patches=self.num_patches, 
                                                      datagen_sampling_pdf=self.datagen_sampling_pdf)
            
            x = np.expand_dims(aux_x, axis=-1)
            y = np.expand_dims(aux_y, axis=-1)

        return x, y


#####
# Pytorch dataset
#####

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
