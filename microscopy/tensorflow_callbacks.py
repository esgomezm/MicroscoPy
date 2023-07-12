from tensorflow.keras.callbacks import Callback
from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np

from . import utils


def tf_normalization(img):
    return (img - tf.math.reduce_min(img)) / (
        tf.math.reduce_max(img) - tf.math.reduce_min(img) + 1e-10
    )


class PerformancePlotCallback(Callback):
    def __init__(self, x_test, y_test, img_saving_path, frequency=1, is_cddpm=False):
        self.x_test = x_test
        self.y_test = y_test
        self.img_saving_path = img_saving_path
        self.frequency = frequency
        self.is_cddpm = is_cddpm

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.frequency == 0:
            if self.is_cddpm:
                y_pred = self.model.predict(self.x_test, self.x_test.shape[0], 500)
            else:
                y_pred = self.model.predict(self.x_test)

            ssim = utils.ssim_loss(self.y_test[0], y_pred[0])

            plt.switch_backend("agg")
            plt.figure(figsize=(15, 5))
            plt.subplot(1, 4, 1)
            plt.title("Input LR image")
            plt.imshow(self.x_test[0], "gray")
            plt.subplot(1, 4, 2)
            plt.title("Ground truth")
            plt.imshow(self.y_test[0], "gray")
            plt.subplot(1, 4, 3)
            plt.title("Prediction")
            plt.imshow(y_pred[0], "gray")
            plt.subplot(1, 4, 4)
            plt.title(f"SSIM: {ssim.numpy():.3f}")
            plt.imshow(1 - tf_normalization(self.y_test[0] - y_pred[0]), "inferno")

            plt.tight_layout()
            plt.savefig(f"{self.img_saving_path}/{epoch}.png")
            plt.close()


class LearningRateObserver(Callback):
    def __init__(self):
        super(LearningRateObserver, self).__init__()
        self.epoch_lrs = []

    def on_epoch_end(self, epoch, logs=None):
        optimizer = self.model.optimizer
        lr = tf.keras.backend.eval(optimizer.lr)
        self.epoch_lrs.append(lr)

    def obtain_lrs(self):
        return self.epoch_lrs


def MultiStepScheduler(
    initial_learning_rate, lr_steps, lr_rate_decay, name="MultiStepLR"
):
    """Multi-steps learning rate scheduler."""
    lr_steps_value = [initial_learning_rate]
    for _ in range(len(lr_steps)):
        lr_steps_value.append(lr_steps_value[-1] * lr_rate_decay)
    return tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries=lr_steps, values=lr_steps_value
    )


class CosineAnnealer:
    def __init__(self, start, end, steps):
        self.start = start
        self.end = end
        self.steps = steps
        self.n = 0

    def step(self):
        self.n += 1
        cos = np.cos(np.pi * (self.n / self.steps)) + 1
        return self.end + (self.start - self.end) / 2.0 * cos


class OneCycleScheduler(Callback):
    """`Callback` that schedules the learning rate on a 1cycle policy as per Leslie Smith's paper(https://arxiv.org/pdf/1803.09820.pdf).
    If the model supports a momentum parameter, it will also be adapted by the schedule.
    The implementation adopts additional improvements as per the fastai library: https://docs.fast.ai/callbacks.one_cycle.html, where
    only two phases are used and the adaptation is done using cosine annealing.
    In phase 1 the LR increases from `lr_max / div_factor` to `lr_max` and momentum decreases from `mom_max` to `mom_min`.
    In the second phase the LR decreases from `lr_max` to `lr_max / (div_factor * 1e4)` and momemtum from `mom_max` to `mom_min`.
    By default the phases are not of equal length, with the phase 1 percentage controlled by the parameter `phase_1_pct`.
    """

    def __init__(
        self,
        lr_max,
        steps,
        mom_min=0.85,
        mom_max=0.95,
        phase_1_pct=0.3,
        div_factor=25.0,
    ):
        super(OneCycleScheduler, self).__init__()
        lr_min = lr_max / div_factor
        final_lr = lr_max / (div_factor * 1e4)
        phase_1_steps = steps * phase_1_pct
        phase_2_steps = steps - phase_1_steps

        self.phase_1_steps = phase_1_steps
        self.phase_2_steps = phase_2_steps
        self.phase = 0
        self.step = 0

        self.phases = [
            [
                CosineAnnealer(lr_min, lr_max, phase_1_steps),
                CosineAnnealer(mom_max, mom_min, phase_1_steps),
            ],
            [
                CosineAnnealer(lr_max, final_lr, phase_2_steps),
                CosineAnnealer(mom_min, mom_max, phase_2_steps),
            ],
        ]

        self.lrs = []
        self.moms = []

    def on_train_begin(self, logs=None):
        self.phase = 0
        self.step = 0

        self.set_lr(self.lr_schedule().start)
        self.set_momentum(self.mom_schedule().start)

    def on_train_batch_begin(self, batch, logs=None):
        self.lrs.append(self.get_lr())
        self.moms.append(self.get_momentum())

    def on_train_batch_end(self, batch, logs=None):
        self.step += 1
        if self.step >= self.phase_1_steps:
            self.phase = 1

        self.set_lr(self.lr_schedule().step())
        self.set_momentum(self.mom_schedule().step())

    def get_lr(self):
        try:
            return tf.keras.backend.get_value(self.model.optimizer.lr)
        except AttributeError:
            return None

    def get_momentum(self):
        try:
            return tf.keras.backend.get_value(self.model.optimizer.momentum)
        except AttributeError:
            return None

    def set_lr(self, lr):
        try:
            tf.keras.backend.set_value(self.model.optimizer.lr, lr)
        except AttributeError:
            pass  # ignore

    def set_momentum(self, mom):
        try:
            tf.keras.backend.set_value(self.model.optimizer.momentum, mom)
        except AttributeError:
            pass  # ignore

    def lr_schedule(self):
        return self.phases[self.phase][0]

    def mom_schedule(self):
        return self.phases[self.phase][1]

    def plot(self):
        ax = plt.subplot(1, 2, 1)
        ax.plot(self.lrs)
        ax.set_title("Learning Rate")
        ax = plt.subplot(1, 2, 2)
        ax.plot(self.moms)
        ax.set_title("Momentum")
