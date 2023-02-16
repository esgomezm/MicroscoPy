import numpy as np
import os
import torch
from torch.utils.data import Dataset
from skimage import img_as_float32
from skimage import io

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from skimage import transform

from utils import hr_to_lr

#### Tensorflow dataset ####

def create_patches(lr_path, hr_path, only_hr, type_hr_data,
                   only_hr_path, down_factor, num_x_patches, 
                   num_y_patches):
    
    ''' Create a list of images patches out of a list of images
    Args:
        lr_path (string): low resolution (LR) image path (input images).
        hr_path (string): high resolution (HR) image path (ground truth images).
        only_hr (boolean): indicates if only HR images will be used.
        type_hr_data (string): in case only HR images would be used, what type of data they would be (Electron microscopy or Fluorescence).
        only_hr_path (string): in case only HR images would be used, HR image path (ground truth images).
        down_factor (int): scale factor between LR and HR images. Example: 2.
        num_x_patches (int): number of patches extracted in the X axis.
        num_y_patches (int): number of patches extracted in the y axis.
        
    Returns:
        list of image patches (LR) and patches of corresponding labels (HR)
    '''
    if only_hr:
        used_hr_path = only_hr_path
    else: 
        used_hr_path = hr_path

    _, extension = os.path.splitext(os.listdir(used_hr_path)[0])
    filenames = [x for x in os.listdir(used_hr_path) if x.endswith(extension)]
    filenames.sort()

    # read training images
    hr_img = img_as_float32(io.imread(os.path.join(used_hr_path, filenames[0])))

    hr_size = hr_img.shape
    
    hr_patch_width = hr_size[0] // num_x_patches
    hr_patch_height = hr_size[1] // num_y_patches

    # Size of the LR images
    lr_patch_width = hr_patch_width // down_factor
    lr_patch_height = hr_patch_height // down_factor

    input_patches = []
    output_patches = []
    
    for n in range(0, len(filenames)):
        if only_hr:
            # If no path to the LR images is given, they will be artificially generated
            lr_img = hr_to_lr(hr_img, down_factor, type_hr_data)
        else:
            lr_img = img_as_float32(io.imread(lr_path + '/' + filenames[n]))

        for i in range(num_x_patches):
            for j in range(num_y_patches):
                output_patches.append(hr_img[i * hr_patch_width : (i+1) * hr_patch_width,
                                       j * hr_patch_height : (j+1) * hr_patch_height])

                input_patches.append(lr_img[i * lr_patch_width : (i+1) * lr_patch_width,
                                      j * lr_patch_height : (j+1) * lr_patch_height])
    return input_patches, output_patches


def create_random_patches( lr_path, hr_path, only_hr, type_hr_data,
                           only_hr_path, down_factor, lr_patch_size_width, 
                           lr_patch_size_height):
    ''' Create a list of images patches out of a list of images
    Args:
        lr_path (string): low resolution (LR) image path (input images).
        hr_path (string): high resolution (HR) image path (ground truth images).
        only_hr (boolean): indicates if only HR images will be used.
        type_hr_data (string): in case only HR images would be used, what type of data they would be (Electron microscopy or Fluorescence).
        only_hr_path (string): in case only HR images would be used, HR image path (ground truth images).
        down_factor (int): scale factor between LR and HR images. Example: 2.
        lr_patch_size_width (int): width of the LR patches.
        lr_patch_size_height (int): height of the LR patches.
        
    Returns:
        list of image patches (LR) and patches of corresponding labels (HR)
    '''
    if only_hr:
        used_hr_path = only_hr_path
    else: 
        used_hr_path = hr_path

    _, extension = os.path.splitext(os.listdir(used_hr_path)[0])
    filenames = [x for x in os.listdir(used_hr_path) if x.endswith(extension)]
    filenames.sort()

    input_patches = []
    output_patches = []
    for n in range(0, len(filenames)):
        hr_img = img_as_float32(io.imread( used_hr_path + '/' + filenames[n]))
        if only_hr:
            # If no path to the LR images is given, they will be artificially generated
            lr_img = hr_to_lr(hr_img, down_factor, type_hr_data)
        else:
            lr_img = img_as_float32(io.imread( lr_path + '/' + filenames[n]))
        
        if lr_patch_size_width is None:
            lr_patch_size_width = lr_img.shape[0]
        if lr_patch_size_height is None:
            lr_patch_size_height = lr_img.shape[1]

        for _ in range((lr_img.shape[0] // lr_patch_size_width)**2):
            lr_idx_width = np.random.randint(0, max(1, lr_img.shape[0] - lr_patch_size_width))
            lr_idx_height = np.random.randint(0, max(1, lr_img.shape[1] - lr_patch_size_height))
            hr_idx_width = lr_idx_width * down_factor
            hr_idx_height = lr_idx_height * down_factor
            
            input_patches.append(lr_img[lr_idx_width : lr_idx_width + lr_patch_size_width,
                                        lr_idx_height : lr_idx_height + lr_patch_size_height])
            output_patches.append(hr_img[hr_idx_width : hr_idx_width + lr_patch_size_width * down_factor,
                                        hr_idx_height : hr_idx_height + lr_patch_size_height * down_factor])

    return input_patches, output_patches

def create_complete_images( lr_path, hr_path, only_hr, 
                           type_hr_data, only_hr_path, down_factor):
    ''' Create a list of images patches out of a list of images
    Args:
        lr_path (string): low resolution (LR) image path (input images).
        hr_path (string): high resolution (HR) image path (ground truth images).
        only_hr (boolean): indicates if only HR images will be used.
        type_hr_data (string): in case only HR images would be used, what type of data they would be (Electron microscopy or Fluorescence).
        only_hr_path (string): in case only HR images would be used, HR image path (ground truth images).
        down_factor (int): scale factor between LR and HR images. Example: 2.
        
    Returns:
        list of image patches (LR) and patches of corresponding labels (HR)
    '''

    return create_random_patches(lr_path, hr_path, only_hr, type_hr_data,
                                only_hr_path, down_factor, None, None)


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
    def __init__(self, 
                 lr_patch_size_x, 
                 lr_patch_size_y,
                 down_factor,
                 transf=None, 
                 validation=False, 
                 validation_split=None,
                 hr_imgs_basedir="", 
                 lr_imgs_basedir="",
                 only_high_resolution_data=False,
                 only_hr_imgs_basedir="",
                 type_of_data="Electron microscopy"):

        if only_high_resolution_data:
            used_hr_imgs_basedir = only_hr_imgs_basedir 
        else: 
            used_hr_imgs_basedir = hr_imgs_basedir

        _, hr_extension = os.path.splitext(os.listdir(used_hr_imgs_basedir)[0])

        hr_filenames = [used_hr_imgs_basedir + '/' + x for x in os.listdir(used_hr_imgs_basedir) if x.endswith(hr_extension)]
        hr_filenames.sort()

        if validation_split is not None:
            val_files = int(len(hr_filenames) * validation_split)
            if validation:
                self.hr_img_names = hr_filenames[:val_files]
            else:
                self.hr_img_names = hr_filenames[val_files:]
        else:
            self.hr_img_names = hr_filenames

        if not only_high_resolution_data:
            _, lr_extension = os.path.splitext(os.listdir(lr_imgs_basedir)[0])

            lr_filenames = [lr_imgs_basedir + '/' + x for x in os.listdir(lr_imgs_basedir) if x.endswith(lr_extension)]
            lr_filenames.sort()

            if validation_split is not None:
                val_lr_files = int(len(lr_filenames) * validation_split)
                if validation:
                    self.lr_img_names = lr_filenames[:val_lr_files]
                else:
                    self.lr_img_names = lr_filenames[val_lr_files:]
            else:
                self.lr_img_names = lr_filenames

        self.only_high_resolution_data = only_high_resolution_data

        self.transf = transf
        self.down_factor = down_factor
        self.type_of_data = type_of_data

        self.lr_patch_size_x = lr_patch_size_x
        self.lr_patch_size_y = lr_patch_size_y
        self.hr_patch_size_x = lr_patch_size_x * down_factor
        self.hr_patch_size_y = lr_patch_size_y * down_factor

        self.lr_shape = io.imread(self.hr_img_names[0]).shape[0]//down_factor

    def __len__(self):
        return len(self.hr_img_names) * (self.lr_shape//self.lr_patch_size_x)**2

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_idx = idx // (self.lr_shape//self.lr_patch_size_x)**2
        
        hr_img = img_as_float32(io.imread(self.hr_img_names[img_idx]))
        
        if self.only_high_resolution_data:
            lr_img = hr_to_lr(hr_img, self.down_factor, self.type_of_data)
        else:
            lr_img = img_as_float32(io.imread(self.lr_img_names[img_idx]))

        lr_idx_x = np.random.randint(0, max(1, lr_img.shape[0] - self.lr_patch_size_x))
        lr_idx_y = np.random.randint(0, max(1, lr_img.shape[1] - self.lr_patch_size_y))

        lr_patch = lr_img[lr_idx_x : lr_idx_x + self.lr_patch_size_x, 
                          lr_idx_y : lr_idx_y + self.lr_patch_size_y]
        lr_patch = lr_patch[:,:,np.newaxis]

        hr_idx_x = lr_idx_x * self.down_factor
        hr_idx_y = lr_idx_y * self.down_factor

        hr_patch = hr_img[hr_idx_x : hr_idx_x + self.hr_patch_size_x, 
                          hr_idx_y : hr_idx_y + self.hr_patch_size_x]

        hr_patch = hr_patch[:,:,np.newaxis]

        sample = {'hr': hr_patch, 'lr': lr_patch}

        if self.transf:
            sample = self.transf(sample)

        return sample
