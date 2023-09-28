# Taken from: https://keras.io/examples/generative/ddim/
 
import tensorflow as tf
import numpy as np
import math
import os

def sinusoidal_embedding(embedding_max_frequency, embedding_dims):
    def sinusoidal_embedding_function(x):
        embedding_min_frequency = 1.0
        frequencies = tf.exp(
            tf.linspace(
                tf.math.log(embedding_min_frequency),
                tf.math.log(embedding_max_frequency),
                embedding_dims // 2,
            )
        )
        angular_speeds = 2.0 * math.pi * frequencies
        embeddings = tf.concat(
            [tf.sin(angular_speeds * x), tf.cos(angular_speeds * x)], axis=3
        )
        return embeddings

    return sinusoidal_embedding_function

def ResidualBlock(width):
    def apply(x):
        input_width = x.shape[3]
        if input_width == width:
            residual = x
        else:
            residual = tf.keras.layers.Conv2D(width, kernel_size=1)(x)
        x = tf.keras.layers.BatchNormalization(center=False, scale=False)(x)
        x = tf.keras.layers.Conv2D(
            width, kernel_size=3, padding="same", activation=tf.keras.activations.swish
        )(x)
        x = tf.keras.layers.Conv2D(width, kernel_size=3, padding="same")(x)
        x = tf.keras.layers.Add()([x, residual])
        return x

    return apply

def DownBlock(width, block_depth):
    def apply(x):
        x, skips = x
        for _ in range(block_depth):
            x = ResidualBlock(width)(x)
            skips.append(x)
        x = tf.keras.layers.AveragePooling2D(pool_size=2)(x)
        return x

    return apply

def UpBlock(width, block_depth):
    def apply(x):
        x, skips = x
        x = tf.keras.layers.UpSampling2D(size=2, interpolation="bilinear")(x)
        for _ in range(block_depth):
            x = tf.keras.layers.Concatenate()([x, skips.pop()])
            x = ResidualBlock(width)(x)
        return x

    return apply

def get_network(
    image_shape, widths, block_depth, embedding_max_frequency, embedding_dims
):
    noisy_images = tf.keras.layers.Input(
        shape=image_shape[:-1] + (image_shape[-1] * 2,)
    )
    noise_variances = tf.keras.layers.Input(shape=(1, 1, 1))

    e = tf.keras.layers.Lambda(
        sinusoidal_embedding(embedding_max_frequency, embedding_dims)
    )(noise_variances)
    e = tf.keras.layers.UpSampling2D(size=image_shape[0], interpolation="nearest")(e)

    x = tf.keras.layers.Conv2D(widths[0], kernel_size=1)(noisy_images)
    x = tf.keras.layers.Concatenate()([x, e])

    skips = []
    for width in widths[:-1]:
        x = DownBlock(width, block_depth)([x, skips])

    for _ in range(block_depth):
        x = ResidualBlock(widths[-1])(x)

    for width in reversed(widths[:-1]):
        x = UpBlock(width, block_depth)([x, skips])

    x = tf.keras.layers.Conv2D(
        image_shape[-1], kernel_size=1, kernel_initializer="zeros"
    )(x)

    return tf.keras.Model([noisy_images, noise_variances], x, name="residual_unet")


class DiffusionModel(tf.keras.Model):
    def __init__(
        self,
        image_shape,
        widths,
        block_depth,
        scale_factor,
        min_signal_rate,
        max_signal_rate,
        batch_size,
        ema,
        embedding_max_frequency,
        embedding_dims,
        verbose=0
    ):
        super().__init__()
        
        self.verbose = verbose
        # image shape must be a tuple with 3 values (h,w,c)
        self.image_shape = (
            image_shape[0] * scale_factor,
            image_shape[1] * scale_factor,
            image_shape[2],
        )

        self.normalizer = tf.keras.layers.Normalization()
        self.network = get_network(
            self.image_shape,
            widths,
            block_depth,
            embedding_max_frequency,
            embedding_dims,
        )
        self.ema_network = tf.keras.models.clone_model(self.network)

        self.min_signal_rate = min_signal_rate
        self.max_signal_rate = max_signal_rate
        self.ema = ema

    def compile(self, **kwargs):
        super().compile(**kwargs)

        if self.verbose > 0:
            print('\n Called compile \n')
            print(f'Eager mode: {tf.executing_eagerly()}')

        self.noise_loss_tracker = tf.keras.metrics.Mean(name="n_loss")
        self.image_loss_tracker = tf.keras.metrics.Mean(name="i_loss")
        
        if self.verbose > 0:
            print('\n End compile \n')

    @property
    def metrics(self):
        return [self.noise_loss_tracker, self.image_loss_tracker]

    def denormalize(self, images):

        if self.verbose > 0:
            print('\n Called denormalize \n')

        # convert the pixel values back to 0-1 range
        denorm_images = self.normalizer.mean + images * self.normalizer.variance**0.5
        clipped_denorm_images = tf.clip_by_value(denorm_images, 0.0, 1.0)

        if self.verbose > 0:
            print(f'images {images.shape} - min: {tf.reduce_max(images)} max: {tf.reduce_max(images)} mean: {tf.reduce_mean(images)}')
            print(f'norm_images {denorm_images.shape} - min: {tf.reduce_max(denorm_images)} max: {tf.reduce_max(denorm_images)} mean: {tf.reduce_mean(denorm_images)}')
            print(f'(return)clipped_denorm_images {clipped_denorm_images.shape} - min: {tf.reduce_max(clipped_denorm_images)} max: {tf.reduce_max(clipped_denorm_images)} mean: {tf.reduce_mean(clipped_denorm_images)}')
        
        if self.verbose > 0:
            print('\n End denormalize \n')

        return clipped_denorm_images

    def diffusion_schedule(self, diffusion_times):

        if self.verbose > 0:
            print('\n Called diffusion_schedule \n')

        # diffusion times -> angles
        start_angle = tf.acos(self.max_signal_rate)
        end_angle = tf.acos(self.min_signal_rate)

        diffusion_angles = start_angle + diffusion_times * (end_angle - start_angle)

        # angles -> signal and noise rates
        signal_rates = tf.cos(diffusion_angles)
        noise_rates = tf.sin(diffusion_angles)
        # note that their squared sum is always: sin^2(x) + cos^2(x) = 1

        if self.verbose > 0:
            print(f'(return) noise_rates {noise_rates.shape} - min: {tf.reduce_max(noise_rates)} max: {tf.reduce_max(noise_rates)} mean: {tf.reduce_mean(noise_rates)}')
            print(f'(return) signal_rates {signal_rates.shape} - min: {tf.reduce_max(signal_rates)} max: {tf.reduce_max(signal_rates)} mean: {tf.reduce_mean(signal_rates)}')
        
        if self.verbose > 0:
            print('\n End diffusion_schedule \n')

        return noise_rates, signal_rates

    def denoise_conditioned(
        self, noisy_images, lr_images, noise_rates, signal_rates, training
    ):
        
        if self.verbose > 0:
            print('\n Called denoise_conditioned \n')
            print(f'Training: {training}')
            print(f'(input) noisy_images {noisy_images.shape} - min: {tf.reduce_max(noisy_images)} max: {tf.reduce_max(noisy_images)} mean: {tf.reduce_mean(noisy_images)}')
            print(f'(input) lr_images {lr_images.shape} - min: {tf.reduce_max(lr_images)} max: {tf.reduce_max(lr_images)} mean: {tf.reduce_mean(lr_images)}')

        # the exponential moving average weights are used at evaluation
        if training:
            if self.verbose > 0:
                print('\n network = self.network \n')
            network = self.network
        else:
            if self.verbose > 0:
                print('\n network = self.ema_network \n')
            network = self.ema_network

        input_data = tf.concat((tf.cast(noisy_images, lr_images.dtype), lr_images), -1)
        
        if self.verbose > 0:
            print(f'input_data {input_data.shape} - min: {tf.reduce_max(input_data)} max: {tf.reduce_max(input_data)} mean: {tf.reduce_mean(input_data)}')

        # predict noise component and calculate the image component using it
        pred_noises = network([input_data, noise_rates**2], training=training)
        pred_images = (noisy_images - noise_rates * pred_noises) / signal_rates

        if self.verbose > 0:
            print(f'(return) noisy_images {noisy_images.shape} - min: {tf.reduce_max(noisy_images)} max: {tf.reduce_max(noisy_images)} mean: {tf.reduce_mean(noisy_images)}')
            print(f'(return) pred_noises {pred_noises.shape} - min: {tf.reduce_max(pred_noises)} max: {tf.reduce_max(pred_noises)} mean: {tf.reduce_mean(pred_noises)}')

        if self.verbose > 0:
            print('\n End denoise_conditioned \n')

        return pred_noises, pred_images

    def reverse_diffusion_conditioned(
        self,
        lr_images,
        initial_noise,
        diffusion_steps
    ):
        
        if self.verbose > 0:
            print('\n Called reverse_diffusion_conditioned \n')
            print(f'(input) lr_images {lr_images.shape} - min: {tf.reduce_max(lr_images)} max: {tf.reduce_max(lr_images)} mean: {tf.reduce_mean(lr_images)}')

        # reverse diffusion = sampling
        num_images = initial_noise.shape[0]
        step_size = 1.0 / diffusion_steps

        # important line:
        # at the first sampling step, the "noisy image" is pure noise
        # but its signal rate is assumed to be nonzero (min_signal_rate)
        next_noisy_images = initial_noise
        for step in range(diffusion_steps):
            if self.verbose > 0:
                print(f'Diffusion step: {step}')

            noisy_images = next_noisy_images

            if self.verbose > 0:
                print(f'(return) noisy_images {noisy_images.shape} - min: {tf.reduce_max(noisy_images)} max: {tf.reduce_max(noisy_images)} mean: {tf.reduce_mean(noisy_images)}')

            # separate the current noisy image to its components
            diffusion_times = tf.ones((num_images, 1, 1, 1)) - step * step_size
            noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
            pred_noises, pred_images = self.denoise_conditioned(
                noisy_images, lr_images, noise_rates, signal_rates, training=False
            )
            # network used in eval mode

            if self.verbose > 0:
                print(f'\t pred_noises {pred_noises.shape} - min: {tf.reduce_max(pred_noises)} max: {tf.reduce_max(pred_noises)} mean: {tf.reduce_mean(pred_noises)}')
                print(f'\t pred_images {pred_images.shape} - min: {tf.reduce_max(pred_images)} max: {tf.reduce_max(pred_images)} mean: {tf.reduce_mean(pred_images)}')

            # remix the predicted components using the next signal and noise rates
            next_diffusion_times = diffusion_times - step_size
            next_noise_rates, next_signal_rates = self.diffusion_schedule(
                next_diffusion_times
            )
            next_noisy_images = (
                next_signal_rates * pred_images + next_noise_rates * pred_noises
            )
            # this new noisy image will be used in the next step

        if self.verbose > 0:
            print(f'(return) pred_images {pred_images.shape} - min: {tf.reduce_max(pred_images)} max: {tf.reduce_max(pred_images)} mean: {tf.reduce_mean(pred_images)}')

        if self.verbose > 0:
            print('\n End reverse_diffusion_conditioned \n')

        return pred_images

    def predict(self, lr_images, diffusion_steps):

        if self.verbose > 0:
            print('\n Called predict \n')
        
        # noise -> images -> denormalized images
        initial_noise = tf.random.normal(shape=(tf.shape(lr_images)[0],) + self.image_shape)
        lr_images = tf.image.resize(lr_images, size=[self.image_shape[0], self.image_shape[1]], method='bicubic')
        generated_images = self.reverse_diffusion_conditioned(
            lr_images, initial_noise, diffusion_steps
        )
        generated_images = self.denormalize(generated_images)
        
        if self.verbose > 0:
            print('\n End predict \n')

        return generated_images

    def train_step(self, data):

        if self.verbose > 0:
            print('\n Called train_step \n')
        
        lr_images, hr_images = data
        
        if self.verbose > 0: 
            print(f'(input) hr_images {hr_images.shape} - min: {tf.reduce_max(hr_images)} max: {tf.reduce_max(hr_images)} mean: {tf.reduce_mean(hr_images)}')
            print(f'(input) lr_images {lr_images.shape} - min: {tf.reduce_max(lr_images)} max: {tf.reduce_max(lr_images)} mean: {tf.reduce_mean(lr_images)}')

        # LR images have to be upscaled to have the same shape as hr_images
        lr_images = tf.image.resize(lr_images, size=[self.image_shape[0], self.image_shape[1]], method='bicubic')
        
        if self.verbose > 0:
            print(f'(input) lr_images {lr_images.shape} - min: {tf.reduce_max(lr_images)} max: {tf.reduce_max(lr_images)} mean: {tf.reduce_mean(lr_images)}')
        
        # normalize images to have standard deviation of 1, like the noises
        hr_images = self.normalizer(hr_images, training=True)

        if self.verbose > 0:
            print(f'(after normalize) hr_images {hr_images.shape} - min: {tf.reduce_max(hr_images)} max: {tf.reduce_max(hr_images)} mean: {tf.reduce_mean(hr_images)}')

        noises = tf.random.normal(shape=(tf.shape(hr_images)[0],) + self.image_shape)

        # sample uniform random diffusion times
        diffusion_times = tf.random.uniform(
            shape=(tf.shape(hr_images)[0], 1, 1, 1), minval=0.0, maxval=1.0
        )
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)

        if self.verbose > 0:
            print(f'(for noisy images) signal_rates {signal_rates.shape} - min: {tf.reduce_max(signal_rates)} max: {tf.reduce_max(signal_rates)} mean: {tf.reduce_mean(signal_rates)}')
            print(f'(for noisy images) hr_images {hr_images.shape} - min: {tf.reduce_max(hr_images)} max: {tf.reduce_max(hr_images)} mean: {tf.reduce_mean(hr_images)}')
            print(f'(for noisy images) noise_rates {noise_rates.shape} - min: {tf.reduce_max(noise_rates)} max: {tf.reduce_max(noise_rates)} mean: {tf.reduce_mean(noise_rates)}')
            print(f'(for noisy images) noises {noises.shape} - min: {tf.reduce_max(noises)} max: {tf.reduce_max(noises)} mean: {tf.reduce_mean(noises)}')
        
        # mix the images with noises accordingly
        noisy_images = tf.math.multiply(signal_rates, hr_images) + tf.math.multiply(noise_rates, noises)

        with tf.GradientTape() as tape:
            # train the network to separate noisy images to their components
            pred_noises, pred_images = self.denoise_conditioned(
                noisy_images, lr_images, noise_rates, signal_rates, training=True
            )

            noise_loss = self.loss(noises, pred_noises)  # used for training
            image_loss = self.loss(hr_images, pred_images)  # only used as metric

        gradients = tape.gradient(noise_loss, self.network.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.network.trainable_weights))

        self.noise_loss_tracker.update_state(noise_loss)
        self.image_loss_tracker.update_state(image_loss)

        # track the exponential moving averages of weights
        for weight, ema_weight in zip(self.network.weights, self.ema_network.weights):
            ema_weight.assign(self.ema * ema_weight + (1 - self.ema) * weight)

        if self.verbose > 0:
            print('\n End train_step \n')

        # KID is not measured during the training phase for computational efficiency
        return {m.name: m.result() for m in self.metrics[:-1]}

    def test_step(self, data):
        
        if self.verbose > 0:
            print('\n Called test_step \n')

        lr_images, hr_images = data

        if self.verbose > 0:
            print(f'(input) hr_images {hr_images.shape} - min: {tf.reduce_max(hr_images)} max: {tf.reduce_max(hr_images)} mean: {tf.reduce_mean(hr_images)}')
            print(f'(input) lr_images {lr_images.shape} - min: {tf.reduce_max(lr_images)} max: {tf.reduce_max(lr_images)} mean: {tf.reduce_mean(lr_images)}')

        lr_images = tf.image.resize(lr_images, size=[self.image_shape[0], self.image_shape[1]], method='bicubic')

        if self.verbose > 0:
            print(f'(after resize) lr_images {lr_images.shape} - min: {tf.reduce_max(lr_images)} max: {tf.reduce_max(lr_images)} mean: {tf.reduce_mean(lr_images)}')

        # normalize images to have standard deviation of 1, like the noises
        hr_images = self.normalizer(hr_images, training=False)
        noises = tf.random.normal(shape=(tf.shape(hr_images)[0],) + self.image_shape)

        # sample uniform random diffusion times
        diffusion_times = tf.random.uniform(
            shape=(tf.shape(hr_images)[0], 1, 1, 1), minval=0.0, maxval=1.0
        )
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
        
        if self.verbose > 0:
            print(f'(for noisy images) signal_rates {signal_rates.shape} - min: {tf.reduce_max(signal_rates)} max: {tf.reduce_max(signal_rates)} mean: {tf.reduce_mean(signal_rates)}')
            print(f'(for noisy images) hr_images {hr_images.shape} - min: {tf.reduce_max(hr_images)} max: {tf.reduce_max(hr_images)} mean: {tf.reduce_mean(hr_images)}')
            print(f'(for noisy images) noise_rates {noise_rates.shape} - min: {tf.reduce_max(noise_rates)} max: {tf.reduce_max(noise_rates)} mean: {tf.reduce_mean(noise_rates)}')
            print(f'(for noisy images) noises {noises.shape} - min: {tf.reduce_max(noises)} max: {tf.reduce_max(noises)} mean: {tf.reduce_mean(noises)}')
        
        # mix the images with noises accordingly
        noisy_images = tf.math.multiply(signal_rates, hr_images) + tf.math.multiply(noise_rates, noises)

        if self.verbose > 0:
            print(f'(noisy_images = signal_rates * hr_images + noise_rates * noises) signal_rates {noisy_images.shape} - min: {tf.reduce_max(noisy_images)} max: {tf.reduce_max(noisy_images)} mean: {tf.reduce_mean(noisy_images)}')

        # use the network to separate noisy images to their components
        pred_noises, pred_images = self.denoise_conditioned(
            noisy_images, lr_images, noise_rates, signal_rates, training=False
        )

        noise_loss = self.loss(noises, pred_noises)
        image_loss = self.loss(hr_images, pred_images)

        self.image_loss_tracker.update_state(image_loss)
        self.noise_loss_tracker.update_state(noise_loss)

        if self.verbose > 0:
            print('\n End test_step \n')

        return {m.name: m.result() for m in self.metrics}

    def load_weights(self, file_path, skip_mismatch=False, by_name=False, options=None):

        file_path_head = os.path.split(file_path)[0]
        file_path_tail = os.path.split(file_path)[1]

        self.network.load_weights(filepath=os.path.join(file_path_head, file_path_tail),
                                  by_name=by_name,
                                  skip_mismatch=skip_mismatch,
                                  options=options,
                                 )
        self.ema_network.load_weights(filepath=os.path.join(file_path_head, 'ema_' + file_path_tail),
                                      by_name=by_name,
                                      skip_mismatch=skip_mismatch,
                                      options=options,
                                     )

        self.normalizer = tf.keras.layers.Normalization(
            mean=np.load(os.path.join(file_path_head, 'mean.npy')), 
            variance=np.load(os.path.join(file_path_head, 'variance.npy'))
        )
        self.normalizer.build(self.image_shape)

    def save_weights(self, file_path, overwrite=True, save_format=None, options=None):

        file_path_head = os.path.split(file_path)[0]
        file_path_tail = os.path.split(file_path)[1]

        self.network.save_weights(filepath=os.path.join(file_path_head, file_path_tail),
                                  overwrite=overwrite,
                                  save_format=save_format,
                                  options=options,
                                 )
        self.ema_network.save_weights(filepath=os.path.join(file_path_head, 'ema_' + file_path_tail),
                                      overwrite=overwrite,
                                      save_format=save_format,
                                      options=options,
                                     )
        
        np.save(os.path.join(file_path_head, 'mean.npy'), self.normalizer.mean.numpy().flatten()[0])
        np.save(os.path.join(file_path_head, 'variance.npy'), self.normalizer.variance.numpy().flatten()[0])
