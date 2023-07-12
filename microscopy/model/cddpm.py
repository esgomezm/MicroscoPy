import tensorflow as tf
import math


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
    ):
        super().__init__()

        # image shape must be a tuple with 3 values (h,w,c)
        self.image_shape = (
            image_shape[0] * scale_factor,
            image_shape[1] * scale_factor,
            image_shape[2],
        )

        self.normalizer = tf.keras.layers.experimental.preprocessing.Normalization()
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
        self.batch_size = batch_size
        self.ema = ema

    def compile(self, **kwargs):
        super().compile(**kwargs)

        self.noise_loss_tracker = tf.keras.metrics.Mean(name="n_loss")
        self.image_loss_tracker = tf.keras.metrics.Mean(name="i_loss")

    @property
    def metrics(self):
        return [self.noise_loss_tracker, self.image_loss_tracker]
        # return [self.noise_loss_tracker, self.image_loss_tracker, self.kid]

    def denormalize(self, images):
        # convert the pixel values back to 0-1 range
        images = self.normalizer.mean + images * self.normalizer.variance**0.5
        return tf.clip_by_value(images, 0.0, 1.0)

    def diffusion_schedule(self, diffusion_times):
        # diffusion times -> angles
        start_angle = tf.acos(self.max_signal_rate)
        end_angle = tf.acos(self.min_signal_rate)

        diffusion_angles = start_angle + diffusion_times * (end_angle - start_angle)

        # angles -> signal and noise rates
        signal_rates = tf.cos(diffusion_angles)
        noise_rates = tf.sin(diffusion_angles)
        # note that their squared sum is always: sin^2(x) + cos^2(x) = 1

        return noise_rates, signal_rates

    def denoise_conditioned(
        self, noisy_images, lr_images, noise_rates, signal_rates, training
    ):
        
        # the exponential moving average weights are used at evaluation
        if training:
            network = self.network
        else:
            network = self.ema_network

        print(f'noisy_images: {noisy_images.shape}')
        print(f'lr_images: {lr_images.shape}')
        input_data = tf.concat((tf.cast(noisy_images, lr_images.dtype), lr_images), -1)

        # predict noise component and calculate the image component using it
        pred_noises = network([input_data, noise_rates**2], training=training)
        pred_images = (noisy_images - noise_rates * pred_noises) / signal_rates

        return pred_noises, pred_images

    def reverse_diffusion_conditioned(
        self,
        lr_images,
        initial_noise,
        diffusion_steps
    ):
        # reverse diffusion = sampling
        num_images = initial_noise.shape[0]
        step_size = 1.0 / diffusion_steps

        # important line:
        # at the first sampling step, the "noisy image" is pure noise
        # but its signal rate is assumed to be nonzero (min_signal_rate)
        next_noisy_images = initial_noise
        for step in range(diffusion_steps):
            noisy_images = next_noisy_images

            # separate the current noisy image to its components
            diffusion_times = tf.ones((num_images, 1, 1, 1)) - step * step_size
            noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
            pred_noises, pred_images = self.denoise_conditioned(
                noisy_images, lr_images, noise_rates, signal_rates, training=False
            )
            # network used in eval mode

            # remix the predicted components using the next signal and noise rates
            next_diffusion_times = diffusion_times - step_size
            next_noise_rates, next_signal_rates = self.diffusion_schedule(
                next_diffusion_times
            )
            next_noisy_images = (
                next_signal_rates * pred_images + next_noise_rates * pred_noises
            )
            # this new noisy image will be used in the next step

        return pred_images

    def predict(self, lr_images, num_images, diffusion_steps):
        # noise -> images -> denormalized images
        initial_noise = tf.random.normal(shape=(num_images,) + self.image_shape)
        lr_images = tf.image.resize(lr_images, size=[self.image_shape[0], self.image_shape[1]])
        generated_images = self.reverse_diffusion_conditioned(
            lr_images, initial_noise, diffusion_steps
        )
        generated_images = self.denormalize(generated_images)
        return generated_images

    def train_step(self, data):
        lr_images, hr_images = data

        # LR images have to be upscaled to have the same shape as hr_images
        lr_images = tf.image.resize(lr_images, size=hr_images[0, :, :, 0].shape)

        # normalize images to have standard deviation of 1, like the noises
        hr_images = self.normalizer(hr_images, training=True)

        noises = tf.random.normal(shape=(self.batch_size,) + self.image_shape)

        # sample uniform random diffusion times
        diffusion_times = tf.random.uniform(
            shape=(self.batch_size, 1, 1, 1), minval=0.0, maxval=1.0
        )
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
        # mix the images with noises accordingly
        noisy_images = signal_rates * hr_images + noise_rates * noises

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

        # KID is not measured during the training phase for computational efficiency
        return {m.name: m.result() for m in self.metrics[:-1]}

    def test_step(self, data):
        lr_images, hr_images = data
        lr_images = tf.image.resize(lr_images, size=hr_images[0, :, :, 0].shape)

        # normalize images to have standard deviation of 1, like the noises
        hr_images = self.normalizer(hr_images, training=False)
        noises = tf.random.normal(shape=(self.batch_size,) + self.image_shape)

        # sample uniform random diffusion times
        diffusion_times = tf.random.uniform(
            shape=(self.batch_size, 1, 1, 1), minval=0.0, maxval=1.0
        )
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
        # mix the images with noises accordingly
        noisy_images = signal_rates * hr_images + noise_rates * noises

        # use the network to separate noisy images to their components
        pred_noises, pred_images = self.denoise_conditioned(
            noisy_images, lr_images, noise_rates, signal_rates, training=False
        )

        noise_loss = self.loss(noises, pred_noises)
        image_loss = self.loss(hr_images, pred_images)

        self.image_loss_tracker.update_state(image_loss)
        self.noise_loss_tracker.update_state(noise_loss)

        return {m.name: m.result() for m in self.metrics}

