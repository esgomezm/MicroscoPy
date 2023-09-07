from tensorflow.keras.callbacks import Callback as tf_callback
from pytorch_lightning.callbacks import Callback as pl_callback
from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np

from . import utils


#####################################
#
# TensorFlow Callbacks.

def tf_normalization(img):
    """
    Normalize the input image using TensorFlow.

    Args:
        img: The input image tensor.

    Returns:
        The normalized image tensor.
    """
    return (img - tf.math.reduce_min(img)) / (
        tf.math.reduce_max(img) - tf.math.reduce_min(img) + 1e-10
    )

class PerformancePlotCallback(tf_callback):
    def __init__(self, x_test, y_test, img_saving_path, frequency=1, is_cddpm=False):
        self.x_test = x_test
        self.y_test = y_test
        self.img_saving_path = img_saving_path
        self.frequency = frequency
        self.is_cddpm = is_cddpm

    def on_epoch_end(self, epoch, logs={}):
        """
        At the end of each epoch (with a frequency) during training, an image of the plot with  
        the LR, HR and prediction is saved.

        Parameters:
            - epoch (int): The current epoch number.
            - logs (dict): Dictionary of logs containing metrics and loss values.

        Returns:
            None
        """
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


class LearningRateObserver(tf_callback):
    def __init__(self):
        super(LearningRateObserver, self).__init__()
        self.epoch_lrs = []

    def on_epoch_end(self, epoch, logs=None):
        """
        At the end of each epoch during training, the learning rate is saved.

        Parameters:
            epoch (int): The current epoch number.
            - logs (dict): Dictionary of logs containing metrics and loss values.

        Returns:
            None
        """
        optimizer = self.model.optimizer
        lr = tf.keras.backend.eval(optimizer.lr)
        self.epoch_lrs.append(lr)

    def obtain_lrs(self):
        """
        Obtain the learning rates for each epoch.

        Returns:
            List[float] or float: The learning rates for each epoch. If the first element of epoch_lrs is of type np.float32, 
            returns the entire list of epoch_lrs. Otherwise, the learning rate scheduler is differnet and the complete list
            of learning rates is in the first element.
        """
        if isinstance(self.epoch_lrs[0], np.float32):
            return self.epoch_lrs
        else:
            return self.epoch_lrs[0]


def MultiStepScheduler(
    initial_learning_rate, lr_steps, lr_rate_decay, name="MultiStepLR"
):
    """
        Creates a multi-step learning rate scheduler for a neural network optimizer.

        Args:
            initial_learning_rate (float): The initial learning rate.
            lr_steps (List[int]): A list of step values at which learning rate decay occurs.
            lr_rate_decay (float): The rate at which the learning rate decays at each step.
            name (str, optional): The name of the scheduler. Defaults to "MultiStepLR".

        Returns:
            tf.keras.optimizers.schedules.PiecewiseConstantDecay: The multi-step learning rate scheduler.

    """
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


class OneCycleScheduler(tf_callback):
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


class CustomCallback(tf_callback):
    def on_train_begin(self, logs=None):
        print("\nStarting training:")
        for k,v in logs.items():
            print("{}: {}".format(k,v))

    def on_train_end(self, logs=None):
        print("\nStop training:")
        for k,v in logs.items():
            print("{}: {}".format(k,v))

    def on_epoch_begin(self, epoch, logs=None):
        print("\nStart epoch {} of training:".format(epoch))
        for k,v in logs.items():
            print("{}: {}".format(k,v))

    def on_epoch_end(self, epoch, logs=None):
        print("\nEnd epoch {} of training:".format(epoch))
        for k,v in logs.items():
            print("{}: {}".format(k,v))

    def on_test_begin(self, logs=None):
        print("\nStart testing:")
        for k,v in logs.items():
            print("{}: {}".format(k,v))

    def on_test_end(self, logs=None):
        print("\nStop testing:")
        for k,v in logs.items():
            print("{}: {}".format(k,v))

    def on_predict_begin(self, logs=None):
        print("\nStart predicting:")
        for k,v in logs.items():
            print("{}: {}".format(k,v))

    def on_predict_end(self, logs=None):
        print("\nStop predicting:")
        for k,v in logs.items():
            print("{}: {}".format(k,v))

    def on_train_batch_begin(self, batch, logs=None):
        print("\n...Training: start of batch {}:".format(batch))
        for k,v in logs.items():
            print("{}: {}".format(k,v))

    def on_train_batch_end(self, batch, logs=None):
        print("\n...Training: end of batch {}:".format(batch))
        for k,v in logs.items():
            print("{}: {}".format(k,v))

    def on_test_batch_begin(self, batch, logs=None):
        print("\n...Testing: start of batch {}:".format(batch))
        for k,v in logs.items():
            print("{}: {}".format(k,v))

    def on_test_batch_end(self, batch, logs=None):
        print("\n...Testing: end of batch {}:".format(batch))
        for k,v in logs.items():
            print("{}: {}".format(k,v))

    def on_predict_batch_begin(self, batch, logs=None):
        print("\n...Predicting: start of batch {}:".format(batch))
        for k,v in logs.items():
            print("{}: {}".format(k,v))

    def on_predict_batch_end(self, batch, logs=None):
        print("\n...Predicting: end of batch {}:".format(batch))
        for k,v in logs.items():
            print("{}: {}".format(k,v))


#
#####################################

#####################################
#
# Pytorch Lightning Callbacks.


class PerformancePlotCallback_Pytorch(pl_callback):
    def __init__(self, x_test, y_test, img_saving_path, frequency=1):
        self.x_test = x_test
        self.y_test = y_test
        self.img_saving_path = img_saving_path
        self.frequency = frequency

    def on_train_epoch_end(self, trainer, pl_module):
        """
        At the end of each epoch (with a frequency) during training, an image of the plot with  
        the LR, HR and prediction is saved.
        
        Args:
            trainer (Trainer): The PyTorch Lightning trainer.
            pl_module (Module): The PyTorch Lightning module.
        Returns:
            None
        """
        if pl_module.current_epoch % self.frequency == 0:
            y_pred = pl_module.forward(self.x_test.cuda()).cpu().detach().numpy()

            print(f'y_pred: {y_pred.shape}')
            print(f'self.x_test: {self.x_test.shape}')
            print(f'self.x_test: {self.x_test.shape}')

            aux_y_pred = np.expand_dims(y_pred[:,0,:,:], axis=-1)
            aux_x_test = np.expand_dims(self.x_test[:,0,:,:], axis=-1)
            aux_y_test = np.expand_dims(self.y_test[:,0,:,:], axis=-1)

            print(f'aux_y_pred: {aux_y_pred.shape}')
            print(f'aux_y_test: {aux_y_test.shape}')
            print(f'aux_x_test: {aux_x_test.shape}')

            ssim = utils.ssim_loss(aux_y_test[0], aux_y_pred[0])

            plt.switch_backend("agg")
            plt.figure(figsize=(15, 5))
            plt.subplot(1, 4, 1)
            plt.title("Input LR image")
            plt.imshow(aux_x_test[0], "gray")
            plt.subplot(1, 4, 2)
            plt.title("Ground truth")
            plt.imshow(aux_y_test[0], "gray")
            plt.subplot(1, 4, 3)
            plt.title("Prediction")
            plt.imshow(aux_y_pred[0], "gray")
            plt.subplot(1, 4, 4)
            plt.title(f"SSIM: {ssim.numpy():.3f}")
            plt.imshow(1 - utils.min_max_normalization(aux_y_test[0] - aux_y_pred[0]), "inferno")

            plt.tight_layout()
            plt.savefig(f"{self.img_saving_path}/{pl_module.current_epoch}.png")
            plt.close()

#
#####################################
