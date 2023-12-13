import numpy as np
import os
from skimage import io
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator

import torch
from torch.utils.data import Dataset
import lightning.pytorch as pl
from torch.utils.data import DataLoader
from torchvision import transforms

from . import crappifiers
from .utils import min_max_normalization as normalization

#####################################
#
# Functions to sample an image using a probability density function.
# Code from: https://github.com/esgomezm/microscopy-dl-suite-tf/blob/fcb8870624208bfb72dc7aea18a90738a081217f/dl-suite/utils

def index_from_pdf(pdf_im):
    """
    Generate the index coordinates from a probability density function (pdf) image.

    Parameters:
    - pdf_im: numpy.ndarray
        The input pdf image.

    Returns:
    - tuple
        A tuple containing the index coordinates (indexh, indexw) of the randomly chosen element from the pdf image.

    Example:
    ```
    pdf_image = np.array([[0.1, 0.2, 0.3],
                          [0.2, 0.3, 0.4],
                          [0.3, 0.4, 0.5]])
    index = index_from_pdf(pdf_image)
    print(index)  # (1, 2)
    ```
    """
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

def sampling_pdf(y, pdf_flag, height, width):
    """
        Generate the function comment for the given function body in a markdown code block with the correct language syntax.

        Parameters:
        - y: the input array
        - pdf_flag: a flag indicating whether to select indexes randomly (0) or based on a PDF (1)
        - height: the height of the crop
        - width: the width of the crop

        Returns:
        - indexh: the index for the center of the crop along the height dimension
        - indexw: the index for the center of the crop along the width dimension
    """

    # Obtain the height and width of the input array
    h, w = y.shape[0], y.shape[1]

    if pdf_flag == 0:
         # If pdf_flag is 0 then select the indexes for the center of the crop randomly
        indexw = np.random.randint(
            np.floor(width // 2),
            max(w - np.floor(width // 2), np.floor(width // 2) + 1),
        )
        indexh = np.random.randint(
            np.floor(height // 2),
            max(h - np.floor(height // 2), np.floor(height // 2) + 1),
        )
    else:
        # If pdf_flag is 1 then select the indexes for the center of the crop based on a PDF

        # crop to fix patch size
        # croped_y = y[int(np.floor(height // 2)):-int(np.floor(height // 2)),
        #              int(np.floor(width // 2)) :-int(np.floor(width // 2))]
        # indexh, indexw = index_from_pdf(croped_y)

        # In order to speed the process, this is done on the Fourier domain
        kernel = np.ones((height, width))

        pdf = np.fft.irfft2(np.fft.rfft2(y) * np.fft.rfft2(kernel, y.shape))
        pdf = normalization(pdf)
        pdf_cropped = pdf[
            min(kernel.shape[0], pdf.shape[0] - 1) :,
            min(kernel.shape[1], pdf.shape[1] - 1) :,
        ]

        indexh, indexw = index_from_pdf(pdf_cropped)
        indexw = indexw + int(np.floor(width // 2))
        indexh = indexh + int(np.floor(height // 2))

    return indexh, indexw

#
#####################################

#####################################
#
# Functions to read image pairs from a given path.

def read_image(file_path, desired_accuracy=np.float32):
    """
    Reads an image from a given file path.

    Args:
        file_path (str): The path to the image file.
        desired_accuracy (type, optional): The desired accuracy of the image. Defaults to np.float32.

    Returns:
        The normalized image.
    """
    return normalization(io.imread(file_path), desired_accuracy=desired_accuracy)


def obtain_scale_factor(hr_filename, lr_filename, scale_factor, crappifier_name):
    """
    Calculates the scale factor between a low-resolution image and a high-resolution image.

    Args:
        hr_filename (str): The path to the high-resolution image file.
        lr_filename (str): The path to the low-resolution image file.
        scale_factor (int): The scale factor to be applied to the low-resolution image.
        crappifier_name (str): The name of the crappifier to use for generating the low-resolution image.

    Raises:
        ValueError: If no scale factor is given and no low-resolution image file is provided.

    Returns:
        int: The scale factor of the images.
    """
    
    if scale_factor is None and lr_filename is None:
        # In case that there is no LR image and no scale factor is given, raise an error
        raise ValueError("A scale factor has to be given.")

    # HR image should always be given, herefore read it
    hr_img = read_image(hr_filename)
    
    if lr_filename is None:
        # If no path to the LR images is given, they will be artificially generated with a crappifier
        lr_img = normalization(
            crappifiers.apply_crappifier(hr_img, scale_factor, crappifier_name)
        )
    else:
        # Otherwise, read the LR image
        lr_img = read_image(lr_filename)

    # Obtain the real scale factor of the image
    images_scale_factor = hr_img.shape[0] // lr_img.shape[0]

    return images_scale_factor


def read_image_pairs(hr_filename, lr_filename, scale_factor, crappifier_name):
    """
    Reads a pair of high-resolution (HR) and low-resolution (LR) images and returns them.

    Parameters:
        hr_filename (str): The path to the HR image file.
        lr_filename (str): The path to the LR image file. If None, the LR image will be artificially generated.
        scale_factor (int): The scale factor for downsampling the LR image.
        crappifier_name (str): The name of the crappifier to be used for generating the LR image.

    Returns:
        tuple: A tuple containing the HR and LR images.
            - hr_img (ndarray): The high-resolution image.
            - lr_img (ndarray): The low-resolution image.
    """
    hr_img = read_image(hr_filename)

    if lr_filename is None:
        # If no path to the LR images is given, they will be artificially generated with a crappifier
        lr_img = normalization(
            crappifiers.apply_crappifier(hr_img, scale_factor, crappifier_name)
        )
    else:
        # Otherwise, read the LR image
        lr_img = read_image(lr_filename)

        # Then calculate the scale factor
        images_scale_factor = hr_img.shape[0] // lr_img.shape[0]

        if scale_factor > images_scale_factor:
            # And in case that the given scale factor is larger than the real scale factor of the images,
            # downsample the low-resolution image to match the given scale factor 
            lr_img = normalization(
                crappifiers.apply_crappifier(
                    lr_img, scale_factor // images_scale_factor, "downsampleonly"
                )
            )

    return hr_img, lr_img

#
#####################################

#####################################
#
# Functions to read images and extract patches from them.

def extract_random_patches_from_image(
    hr_filename,
    lr_filename,
    scale_factor,
    crappifier_name,
    lr_patch_shape,
    datagen_sampling_pdf,
    verbose = 0
):
    """
    Extracts random patches from an image.

    :param hr_filename: The path to the high-resolution image file.
    :param lr_filename: The path to the low-resolution image file.
    :param scale_factor: The scale factor used for downsampling the image.
    :param crappifier_name: The name of the crappifier used for generating the low-resolution image.
    :param lr_patch_shape: The shape of the patches in the low-resolution image. If None, the complete image will be used.
    :param datagen_sampling_pdf: A flag indicating whether a probability density function (PDF) is used for sampling the patch coordinates.
    :return: A tuple containing the low-resolution and high-resolution patches.
    :raises ValueError: If the patch size is bigger than the given images.
    """

    # First lets read the images from given paths
    hr_img, lr_img = read_image_pairs(
        hr_filename, lr_filename, scale_factor, crappifier_name
    )

    if lr_patch_shape is None:
        # In case that the patch shape (on the low-resolution image) is not given, 
        # the complete image will be used
        lr_patch_size_width = lr_img.shape[0]
        lr_patch_size_height = lr_img.shape[1]
    else:
        # Otherwise, use the given patch shape
        lr_patch_size_width = lr_patch_shape[0]
        lr_patch_size_height = lr_patch_shape[1]

    if (
        lr_img.shape[0] < lr_patch_size_width
        or hr_img.shape[0] < lr_patch_size_width * scale_factor
    ):
        # In case that the patch size is bigger than the given images, raise an error
        raise ValueError("Patch size is bigger than the given images.")

    if (
        lr_patch_size_width == lr_img.shape[0]
        and lr_patch_size_height == lr_img.shape[1]
    ):
        # In case that the patch size is the same as the given images, return the images
        lr_patch = lr_img
        hr_patch = hr_img
    else:
        # Otherwise, extract the patch 

        # For that the indexes for the center of the patch are calculated (using a PDF or ranfomly)
        lr_idx_width, lr_idx_height = sampling_pdf(
            y=lr_img,
            pdf_flag=datagen_sampling_pdf,
            height=lr_patch_size_height,
            width=lr_patch_size_width,
        )

        # Calculate the lower-row (lr) and upper-row (ur) coordinates
        lr = int(lr_idx_height - np.floor(lr_patch_size_height // 2))
        ur = int(lr_idx_height + np.round(lr_patch_size_height // 2))

        # Calculate the lower-column (lc) and upper-column (uc) coordinates
        lc = int(lr_idx_width - np.floor(lr_patch_size_width // 2))
        uc = int(lr_idx_width + np.round(lr_patch_size_width // 2))

        # Extract the patches
        lr_patch = lr_img[lc:uc, lr:ur]
        hr_patch = hr_img[
            lc * scale_factor : uc * scale_factor, lr * scale_factor : ur * scale_factor
        ]

    if verbose > 3:
        print('\nExtracting patches:')
        print("lr_patch[{}:{}, {}:{}] - {} - min: {} max: {}".format(lc, uc, lr, ur, lr_patch.shape,
                                                                lr_patch.min(), lr_patch.max()))
        print(lr_filename)
        print(f'\tLR_patch: {lr_patch[0,:5]}')
        print(f'\t{lr_img[0,:5]}')
        print("hr_patch[{}:{}, {}:{}] - {} - min: {} max: {}".format(lc * scale_factor, uc * scale_factor, 
                                              lr * scale_factor, ur * scale_factor, hr_patch.shape,
                                              hr_patch.min(), hr_patch.max()))
        print(f'\t{hr_patch[0,:5]}')
        print(f'\t{hr_img[0,:5]}')
        print(hr_filename)

    return lr_patch, hr_patch


def extract_random_patches_from_folder(
    hr_data_path,
    lr_data_path,
    filenames,
    scale_factor,
    crappifier_name,
    lr_patch_shape,
    datagen_sampling_pdf,
    verbose = 0
):
    """
    Extracts random patches from a folder of high-resolution and low-resolution images.
    
    Args:
        hr_data_path (str): The path to the folder containing the high-resolution images.
        lr_data_path (str): The path to the folder containing the low-resolution images.
        filenames (list): A list of filenames of the images to extract patches from.
        scale_factor (float): The scale factor for downsampling the images.
        crappifier_name (str): The name of the crappifier to use for downsampling.
        lr_patch_shape (tuple): The shape of the low-resolution patches to extract.
        datagen_sampling_pdf (str): The probability density function for sampling the patches.
    
    Returns:
        final_lr_patches (numpy.ndarray): An array of extracted low-resolution patches.
        final_hr_patches (numpy.ndarray): An array of extracted high-resolution patches.
        actual_scale_factor (float): The actual scale factor used for downsampling.
    """
    
    # First lets check what is the scale factor, in case None is given
    actual_scale_factor = obtain_scale_factor(
        hr_filename=os.path.join(hr_data_path, filenames[0]),
        lr_filename=None
        if lr_data_path is None
        else os.path.join(lr_data_path, filenames[0]),
        scale_factor=scale_factor,
        crappifier_name=crappifier_name,
    )

    final_lr_patches = []
    final_hr_patches = []

    # Then for a fiven list of filenames, extract a single patch for each image
    for f in filenames:
        hr_image_path = os.path.join(hr_data_path, f)
        if lr_data_path is not None:
            lr_image_path = os.path.join(lr_data_path, f)
        else:
            lr_image_path = None
        lr_patches, hr_patches = extract_random_patches_from_image(
            hr_image_path,
            lr_image_path,
            actual_scale_factor,
            crappifier_name,
            lr_patch_shape,
            datagen_sampling_pdf,
            verbose=verbose
        )
        final_lr_patches.append(lr_patches)
        final_hr_patches.append(hr_patches)

    final_lr_patches = np.array(final_lr_patches)
    final_hr_patches = np.array(final_hr_patches)

    return final_lr_patches, final_hr_patches, actual_scale_factor

#
#####################################

#####################################
#
# Functions to define a TensorFlow datasets with its generator.

class TFDataGenerator:
    def __init__(
        self,
        filenames,
        hr_data_path,
        lr_data_path,
        scale_factor,
        crappifier_name,
        lr_patch_shape,
        datagen_sampling_pdf,
        validation_split,
        verbose
    ):
        self.filenames = np.array(filenames)
        self.indexes = np.arange(len(self.filenames))

        self.hr_data_path = hr_data_path
        self.lr_data_path = lr_data_path
        self.scale_factor = scale_factor
        self.crappifier_name = crappifier_name
        self.lr_patch_shape = lr_patch_shape
        self.datagen_sampling_pdf = datagen_sampling_pdf

        self.validation_split = validation_split      
    
        # In order to not calculate the actual scale factor on each step, its calculated on the initialization  
        _, _, actual_scale_factor = extract_random_patches_from_folder(
                                        self.hr_data_path,
                                        self.lr_data_path,
                                        [self.filenames[0]],
                                        scale_factor=self.scale_factor,
                                        crappifier_name=self.crappifier_name,
                                        lr_patch_shape=self.lr_patch_shape,
                                        datagen_sampling_pdf=self.datagen_sampling_pdf,
                                    )
        self.actual_scale_factor = actual_scale_factor
        self.verbose = verbose

    def __len__(self):
        """
        Returns the length of the object.
        Which will be used for the number of images on each epoch (not the batches).

        :return: int
            The length of the object.
        """
        return int(len(self.filenames))

    def __getitem__(self, idx):
        """
        Retrieves a pair of low-resolution and high-resolution image patches from the dataset.

        Parameters:
            idx (int): The index of the image in the dataset.

        Returns:
            tuple: A tuple containing the low-resolution and high-resolution image patches.
                - lr_patches (ndarray): A 4D numpy array of low-resolution image patches.
                - hr_patches (ndarray): A 4D numpy array of high-resolution image patches.
        """
        hr_image_path = os.path.join(self.hr_data_path, self.filenames[idx])
        if self.lr_data_path is not None:
            lr_image_path = os.path.join(self.lr_data_path, self.filenames[idx])
        else:
            lr_image_path = None

        if self.verbose > 3:
            print('Extracting patches for image {}'.format(os.path.join(self.hr_data_path, self.filenames[idx])))

        aux_lr_patches, aux_hr_patches = extract_random_patches_from_image(
            hr_image_path,
            lr_image_path,
            self.actual_scale_factor,
            self.crappifier_name,
            self.lr_patch_shape,
            self.datagen_sampling_pdf,
            verbose=self.verbose
        )

        lr_patches = np.expand_dims(aux_lr_patches, axis=-1)
        hr_patches = np.expand_dims(aux_hr_patches, axis=-1)

        return lr_patches, hr_patches

    def __call__(self):
        """
        Calls the object as a function.

        Yields each item in the object by iterating over it.
        """
        for i in range(self.__len__()):
            yield self.__getitem__(i)


def prerpoc_func(x, y, rotation, horizontal_flip, vertical_flip):
    """
    Applies random preprocessing transformations to the input images.

    Args:
        x (Tensor): The input image tensor.
        y (Tensor): The target image tensor.
        rotation (bool): Whether to apply rotation.
        horizontal_flip (bool): Whether to apply horizontal flip.
        vertical_flip (bool): Whether to apply vertical flip.

    Returns:
        Tuple[Tensor, Tensor]: The preprocessed input and target image tensors.
    """
    apply_rotation = (tf.random.uniform(shape=[]) < 0.5) and rotation
    apply_horizontal_flip = (tf.random.uniform(shape=[]) < 0.5) and horizontal_flip
    apply_vertical_flip = (tf.random.uniform(shape=[]) < 0.5) and vertical_flip

    if apply_rotation:
        rotation_times = np.random.randint(0, 5)
        x = tf.image.rot90(x, rotation_times)
        y = tf.image.rot90(y, rotation_times)
    if apply_horizontal_flip:
        x = tf.image.flip_left_right(x)
        y = tf.image.flip_left_right(y)
    if apply_vertical_flip:
        x = tf.image.flip_up_down(x)
        y = tf.image.flip_up_down(y)
    return x, y


def TFDataset(
    filenames,
    hr_data_path,
    lr_data_path,
    scale_factor,
    crappifier_name,
    lr_patch_shape,
    datagen_sampling_pdf,
    validation_split,
    batch_size,
    rotation,
    horizontal_flip,
    vertical_flip,
    verbose
):
    """
    Generate a TensorFlow Dataset for training and validation.

    Args:
        filenames (list): List of filenames to be used for generating the dataset.
        hr_data_path (str): Path to the high-resolution data directory.
        lr_data_path (str): Path to the low-resolution data directory.
        scale_factor (int): Scale factor for upsampling the low-resolution data.
        crappifier_name (str): Name of the crappifier to be used for generating low-resolution data.
        lr_patch_shape (tuple): Shape of the low-resolution patches.
        datagen_sampling_pdf (str): Path to the sampling PDF file for data generation.
        validation_split (float): Proportion of data to be used for validation.
        batch_size (int): Number of samples per batch.
        rotation (bool): Whether to apply random rotations to the data.
        horizontal_flip (bool): Whether to apply random horizontal flips to the data.
        vertical_flip (bool): Whether to apply random vertical flips to the data.

    Returns:
        tuple: A tuple containing the following elements:
            - dataset (tf.data.Dataset): The generated TensorFlow Dataset.
            - lr_shape (tuple): Shape of the low-resolution data.
            - hr_shape (tuple): Shape of the high-resolution data.
            - actual_scale_factor (float): The actual scale factor used for upsampling.
    """
    data_generator = TFDataGenerator(
        filenames=filenames,
        hr_data_path=hr_data_path,
        lr_data_path=lr_data_path,
        scale_factor=scale_factor,
        crappifier_name=crappifier_name,
        lr_patch_shape=lr_patch_shape,
        datagen_sampling_pdf=datagen_sampling_pdf,
        validation_split=validation_split,
        verbose=verbose
    )

    # Get the first item to extract information from it
    lr, hr = data_generator.__getitem__(0)
    actual_scale_factor = data_generator.actual_scale_factor

    # Create the dataset generator
    dataset = tf.data.Dataset.from_generator(
        data_generator,
        output_types=(lr.dtype, hr.dtype),
        output_shapes=(tf.TensorShape(lr.shape), tf.TensorShape(hr.shape)),
    )

    # Map the preprocessing function
    dataset = dataset.map(
        lambda x, y: prerpoc_func(x, y, rotation, horizontal_flip, vertical_flip),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    # Batch the data
    dataset = dataset.batch(batch_size)

    return (
        dataset,
        (data_generator.__len__(),) + (lr.shape),
        (data_generator.__len__(),) + (hr.shape),
        actual_scale_factor,
    )

#
#####################################

#####################################
#
# Functions to define a different Tensorflow Data generator which is based on Sequence.

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(
        self,
        filenames,
        hr_data_path,
        lr_data_path,
        scale_factor,
        crappifier_name,
        lr_patch_shape,
        datagen_sampling_pdf,
        validation_split,
        batch_size,
        rotation,
        horizontal_flip,
        vertical_flip,
        shuffle=True,
    ):
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
        self.datagen_sampling_pdf = datagen_sampling_pdf

        self.validation_split = validation_split
        self.batch_size = batch_size

        self.rotation = rotation
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip

        # Make an initial shuffle
        self.shuffle = shuffle
        self.on_epoch_end()

        # In order to not calculate the actual scale factor on each step, its calculated on the initialization  
        _, _, actual_scale_factor = extract_random_patches_from_folder(
                                        self.hr_data_path,
                                        self.lr_data_path,
                                        [self.filenames[0]],
                                        scale_factor=self.scale_factor,
                                        crappifier_name=self.crappifier_name,
                                        lr_patch_shape=self.lr_patch_shape,
                                        datagen_sampling_pdf=self.datagen_sampling_pdf,
                                    )
        self.actual_scale_facto

    def __len__(self):
        """
        Denotes the number of batches per epoch.

        Returns:
            int: The number of batches per epoch.
        """
        return int(np.floor(len(self.filenames) / self.batch_size))

    def get_sample(self, idx):
        """
        Get a sample from the dataset.

        Parameters:
            idx (int): The index of the sample to retrieve.

        Returns:
            tuple: A tuple containing the x and y values of the sample, as well as the actual scale factor.
        """
        x, y = self.__getitem__(idx)

        return x, y, self.actual_scale_factor

    def on_epoch_end(self):
        """
        Perform actions at the end of each epoch.

        This method is called at the end of each epoch in the training process.
        It updates the `indexes` attribute by creating a numpy array of indices
        corresponding to the length of the `filenames` attribute. If `shuffle`
        is set to `True`, it shuffles the indices using the `np.random.shuffle`
        function.
        """
        self.indexes = np.arange(len(self.filenames))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        """
        Returns a tuple of low-resolution patches and high-resolution patches corresponding to the given index.

        Parameters:
            index (int): The index of the batch.
        
        Returns:
            tuple: A tuple containing the low-resolution patches and high-resolution patches.
        """
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        # Find list of IDs
        list_IDs_temp = [self.indexes[k] for k in indexes]
        # Generate data
        lr_patches, hr_patches = self.__data_generation(list_IDs_temp)
        return lr_patches, hr_patches

    def __call__(self):
        """
        Calls the object as a function.

        Yields each item in the object by iterating over it.
        """
        for i in range(self.__len__()):
            yield self.__getitem__(i)

    def __preprocess(self, x, y):
        """
        Preprocesses the input data by applying random rotations, horizontal flips, and vertical flips.

        Parameters:
            x (ndarray): The input data to be preprocessed.
            y (ndarray): The target data to be preprocessed.

        Returns:
            processed_x (ndarray): The preprocessed input data.
            processed_y (ndarray): The preprocessed target data.
        """
        apply_rotation = (np.random.random() < 0.5) * self.rotation
        apply_horizontal_flip = (np.random.random() < 0.5) * self.horizontal_flip
        apply_vertical_flip = (np.random.random() < 0.5) * self.vertical_flip

        processed_x = np.copy(x)
        processed_y = np.copy(y)

        if apply_rotation:
            rotation_times = np.random.randint(0, 5)
            processed_x = np.rot90(processed_x, rotation_times, axes=(1, 2))
            processed_y = np.rot90(processed_y, rotation_times, axes=(1, 2))
        if apply_horizontal_flip:
            processed_x = np.flip(processed_x, axis=2)
            processed_y = np.flip(processed_y, axis=2)
        if apply_vertical_flip:
            processed_x = np.flip(processed_x, axis=1)
            processed_y = np.flip(processed_y, axis=1)

        return processed_x, processed_y

    def __data_generation(self, list_IDs_temp):
        """
        Generate data batches for training or validation.

        Parameters:
            list_IDs_temp (list): The list of data IDs to generate the data for.

        Returns:
            lr_patches (ndarray): The low-resolution image patches generated from the data.
            hr_patches (ndarray): The high-resolution image patches generated from the data.
        """
        
        hr_image_path = os.path.join(self.hr_data_path, self.filenames[idx])
        if self.lr_data_path is not None:
            lr_image_path = os.path.join(self.lr_data_path, self.filenames[idx])
        else:
            lr_image_path = None

        aux_lr_patches, aux_hr_patches = extract_random_patches_from_image(
            hr_image_path,
            lr_image_path,
            self.actual_scale_factor,
            self.crappifier_name,
            self.lr_patch_shape,
            self.datagen_sampling_pdf,
        )

        lr_patches = np.expand_dims(aux_lr_patches, axis=-1)
        hr_patches = np.expand_dims(aux_hr_patches, axis=-1)

        lr_patches, hr_patches = self.__preprocess(lr_patches, hr_patches)

        return lr_patches, hr_patches

#
#####################################

#####################################
#
# Functions that will be used to define and create an old version of the TensorFlow dataset.
# This version would load the complete dataset on memory and only once, therefore
# it would have the same images for each epoch. Even if is may be faster, 

from skimage import transform

def random_90rotation( img ):
    """
    Rotate an image randomly by 90 degrees.

    Parameters:
        img (array-like): The image to be rotated.

    Returns:
        array-like: The rotated image.
    """
    return transform.rotate(img, 90*np.random.randint( 0, 5 ), preserve_range=True)


from tensorflow.keras.preprocessing.image import ImageDataGenerator

def get_train_val_generators(X_data, Y_data, batch_size=32, seed=42, show_examples=False):
    """
    Generate train and validation data generators using image data augmentation.

    :param X_data: The input data for training.
    :param Y_data: The target data for training.
    :param batch_size: The batch size used for training. Default is 32.
    :param seed: The seed used for random number generation. Default is 42.
    :param show_examples: Whether to show examples of augmented images. Default is False.
    :return: The train generator that yields augmented image and target data.
    """

    # Image data generator distortion options
    data_gen_args = dict( #rotation_range = 45,
                          #width_shift_range=0.2,
                          #height_shift_range=0.2,
                          #shear_range=0.2,
                          #brightness_range=[1., 1.],
                          #rescale=1./255,
                          preprocessing_function=random_90rotation,
                          horizontal_flip=True,
                          vertical_flip=True,
                          fill_mode='reflect')


    # Train data, provide the same seed and keyword arguments to the fit and flow methods
    X_datagen = ImageDataGenerator(**data_gen_args)
    Y_datagen = ImageDataGenerator(**data_gen_args)
    X_datagen.fit(X_data, augment=True, seed=seed)
    Y_datagen.fit(Y_data, augment=True, seed=seed)
    X_data_augmented = X_datagen.flow(X_data, batch_size=batch_size, shuffle=True, seed=seed)
    Y_data_augmented = Y_datagen.flow(Y_data, batch_size=batch_size, shuffle=True, seed=seed)

    # combine generators into one which yields image and masks
    train_generator = zip(X_data_augmented, Y_data_augmented)

    return train_generator

#
#####################################

#####################################
#
# Functions that will be used to define and create the Pytorch dataset

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        hr, lr = sample["hr"], sample["lr"]
        # Pytorch is (batch, channels, width, height)
        hr = hr.transpose((2, 0, 1))
        lr = lr.transpose((2, 0, 1))
        return {"hr": torch.from_numpy(hr), "lr": torch.from_numpy(lr)}

class RandomHorizontalFlip(object):
    """Random horizontal flip"""
    def __init__(self):
        self.rng = np.random.default_rng()

    def __call__(self, sample):
        hr, lr = sample["hr"], sample["lr"]

        if self.rng.random() < 0.5:
            hr = np.flip(hr, 1)
            lr = np.flip(lr, 1)

        return {"hr": hr.copy(), "lr": lr.copy()}

class RandomVerticalFlip(object):
    """Random vertical flip"""
    def __init__(self):
        self.rng = np.random.default_rng()

    def __call__(self, sample):
        hr, lr = sample["hr"], sample["lr"]

        if self.rng.random() < 0.5:
            hr = np.flip(hr, 0)
            lr = np.flip(lr, 0)

        return {"hr": hr.copy(), "lr": lr.copy()}

class RandomRotate(object):
    """Random rotation"""
    def __init__(self):
        self.rng = np.random.default_rng()

    def __call__(self, sample):
        hr, lr = sample["hr"], sample["lr"]

        k = self.rng.integers(4)

        hr = np.rot90(hr, k=k)
        lr = np.rot90(lr, k=k)

        return {"hr": hr.copy(), "lr": lr.copy()}


class PytorchDataset(Dataset):
    """Pytorch's Dataset type object used to obtain the train and
    validation information during the training process. Saves the
    filenames as an attribute and only loads the ones rquired for
    the training batch, reducing the required RAM memory during
    and after the training.
    """

    def __init__(
        self,
        hr_data_path,
        lr_data_path,
        filenames,
        scale_factor,
        crappifier_name,
        lr_patch_shape,
        transformations,
        datagen_sampling_pdf,
        val_split=None,
        validation=False,
        verbose=0
    ):
        self.hr_data_path = hr_data_path
        self.lr_data_path = lr_data_path

        if val_split is None:
            self.filenames = filenames
        elif validation:
            self.filenames = filenames[: int(val_split * len(filenames))]
        else:
            self.filenames = filenames[int(val_split * len(filenames)) :]

        self.transformations = transformations
        self.scale_factor = scale_factor
        self.crappifier_name = crappifier_name
        self.lr_patch_shape = lr_patch_shape

        self.datagen_sampling_pdf = datagen_sampling_pdf

        # In order to not calculate the actual scale factor on each step, its calculated on the initialization  
        _, _, actual_scale_factor = extract_random_patches_from_folder(
                                        self.hr_data_path,
                                        self.lr_data_path,
                                        [self.filenames[0]],
                                        scale_factor=self.scale_factor,
                                        crappifier_name=self.crappifier_name,
                                        lr_patch_shape=self.lr_patch_shape,
                                        datagen_sampling_pdf=self.datagen_sampling_pdf,
                                    )
        self.actual_scale_factor = actual_scale_factor

        self.verbose = verbose

    def __len__(self):
        """
        Returns the length of the object.
        Which will be used for the number of images on each epoch (not the batches).

        :return: int
            The length of the object.
        """
        return len(self.filenames)

    def __getitem__(self, idx):
        """
        Returns a sample from the dataset.

        Parameters:
            - idx (int): The index of the sample to retrieve.

        Returns:
            - sample (dict): A dictionary containing the high-resolution and low-resolution patches of an image.
                - hr (ndarray): The high-resolution patches.
                - lr (ndarray): The low-resolution patches.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        hr_image_path = os.path.join(self.hr_data_path, self.filenames[idx])
        if self.lr_data_path is not None:
            lr_image_path = os.path.join(self.lr_data_path, self.filenames[idx])
        else:
            lr_image_path = None

        aux_lr_patches, aux_hr_patches = extract_random_patches_from_image(
            hr_image_path,
            lr_image_path,
            self.actual_scale_factor,
            self.crappifier_name,
            self.lr_patch_shape,
            self.datagen_sampling_pdf,
            verbose=self.verbose
        )

        lr_patches = np.expand_dims(aux_lr_patches, axis=-1)
        hr_patches = np.expand_dims(aux_hr_patches, axis=-1)

        sample = {"hr": hr_patches, "lr": lr_patches}

        if self.transformations:
            sample = self.transformations(sample)

        if self.verbose > 3:
            print('__get_item__')
            print(sample)

        return sample

class PytorchDataModuler(pl.LightningDataModule):
    def __init__(
            self, 
            lr_patch_size_x: int = 128,
            lr_patch_size_y: int = 128,
            batch_size: int = 8,
            scale_factor: int = 2,
            datagen_sampling_pdf: int = 1,
            rotation: bool = True,
            horizontal_flip: bool = True,
            vertical_flip: bool = True,
            train_hr_path: str = "",
            train_lr_path: str = "",
            train_filenames: list = [],
            val_hr_path: str = "",
            val_lr_path: str = "",
            val_filenames: list = [],
            test_hr_path: str = "",
            test_lr_path: str = "",
            test_filenames: list = [],
            crappifier_method: str = "downsampleonly",
            verbose: int = 0,
            ):
        #Define required parameters here
        super().__init__()

        self.lr_patch_size_x = lr_patch_size_x 
        self.lr_patch_size_y = lr_patch_size_y
        self.batch_size = batch_size
        self.scale_factor = scale_factor
        self.datagen_sampling_pdf = datagen_sampling_pdf

        self.rotation = rotation
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip

        self.train_hr_path = train_hr_path
        self.train_lr_path = train_lr_path
        self.train_filenames = train_filenames

        self.val_hr_path = val_hr_path
        self.val_lr_path = val_lr_path
        self.val_filenames = val_filenames

        self.test_hr_path = test_hr_path
        self.test_lr_path = test_lr_path
        self.test_filenames = test_filenames

        self.crappifier_method = crappifier_method

        self.verbose = verbose

    
    def prepare_data(self):
        # Define steps that should be done
        # on only one GPU, like getting data.
        pass
    
    def setup(self, stage=None):
        # Define steps that should be done on 
        # every GPU, like splitting data, applying
        # transform etc.

        print(f'Dataset setup stage: {stage}')

        # Assign Train/val split(s) for use in Dataloaders
        if stage == "fit":
            train_transformations = []

            if self.horizontal_flip:
                train_transformations.append(RandomHorizontalFlip())
            if self.vertical_flip:
                train_transformations.append(RandomVerticalFlip())
            if self.rotation:
                train_transformations.append(RandomRotate())

            train_transformations.append(ToTensor())

            train_transf = transforms.Compose(train_transformations)
            val_transf = ToTensor()
            
            if self.val_hr_path is None:
                self.train_dataset = PytorchDataset(
                    hr_data_path=self.train_hr_path,
                    lr_data_path=self.train_lr_path,
                    filenames=self.train_filenames,
                    scale_factor=self.scale_factor,
                    crappifier_name=self.crappifier_method,
                    lr_patch_shape=(
                        self.lr_patch_size_x,
                        self.lr_patch_size_y,
                    ),
                    transformations=train_transf,
                    datagen_sampling_pdf=self.datagen_sampling_pdf,
                    val_split=0.1,
                    validation=False,
                    verbose=self.verbose
                )

                self.val_dataset = PytorchDataset(
                    hr_data_path=self.train_hr_path,
                    lr_data_path=self.train_lr_path,
                    filenames=self.train_filenames,
                    scale_factor=self.scale_factor,
                    crappifier_name=self.crappifier_method,
                    lr_patch_shape=(
                        self.lr_patch_size_x,
                        self.lr_patch_size_y,
                    ),
                    transformations=val_transf,
                    datagen_sampling_pdf=self.datagen_sampling_pdf,
                    val_split=0.1,
                    validation=True,
                    verbose=self.verbose
                )

            else:
                self.train_dataset = PytorchDataset(
                    hr_data_path=self.train_hr_path,
                    lr_data_path=self.train_lr_path,
                    filenames=self.train_filenames,
                    scale_factor=self.scale_factor,
                    crappifier_name=self.crappifier_method,
                    lr_patch_shape=(
                        self.lr_patch_size_x,
                        self.lr_patch_size_y,
                    ),
                    transformations=train_transf,
                    datagen_sampling_pdf=self.datagen_sampling_pdf,
                    verbose=self.verbose
                )
                
                self.val_dataset = PytorchDataset(
                    hr_data_path=self.val_hr_path,
                    lr_data_path=self.val_lr_path,
                    filenames=self.val_filenames,
                    scale_factor=self.scale_factor,
                    crappifier_name=self.crappifier_method,
                    lr_patch_shape=(
                        self.lr_patch_size_x,
                        self.lr_patch_size_y,
                    ),
                    transformations=val_transf,
                    datagen_sampling_pdf=self.datagen_sampling_pdf,
                    verbose=self.verbose
                )

        if stage == "test":        
            self.test_dataset = PytorchDataset(
                hr_data_path=self.test_hr_path,
                lr_data_path=self.test_lr_path,
                filenames=self.test_filenames,
                scale_factor=self.scale_factor,
                crappifier_name=self.crappifier_method,
                lr_patch_shape=None,
                transformations=ToTensor(),
                datagen_sampling_pdf=self.datagen_sampling_pdf,
            )

        if stage == "predict":        
            # Is the same as the test_dataset but it also needs to be defined
            self.predict_dataset = PytorchDataset(
                hr_data_path=self.test_hr_path,
                lr_data_path=self.test_lr_path,
                filenames=self.test_filenames,
                scale_factor=self.scale_factor,
                crappifier_name=self.crappifier_method,
                lr_patch_shape=None,
                transformations=ToTensor(),
                datagen_sampling_pdf=self.datagen_sampling_pdf,
            )
    
    def train_dataloader(self):
        # Return DataLoader for Training Data here
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=32)
    
    def val_dataloader(self):
        # Return DataLoader for Validation Data here
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=32)
    
    def test_dataloader(self):
        # Return DataLoader for Testing Data here
        return DataLoader(self.test_dataset, batch_size=1, num_workers=32)

    def predict_dataloader(self):
        # Return DataLoader for Predicting Data here        
        return DataLoader(self.predict_dataset, batch_size=1, num_workers=32)

    def teardown(self, stage):
        if stage == "fit":
            del self.train_dataset
            del self.val_dataset
        if stage == "test":        
            del self.test_dataset
        if stage == "predict":        
            del self.predict_dataset
#
#####################################