import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.callbacks import ReduceLROnPlateau

import torch

from . import tensorflow_callbacks


def select_optimizer(
    library_name,
    optimizer_name,
    learning_rate,
    check_point,
    parameters,
    additional_configuration,
):
    if library_name == "tensorflow":
        return select_tensorflow_optimizer(
            optimizer_name=optimizer_name,
            learning_rate=learning_rate,
            additional_configuration=additional_configuration,
        )
    elif library_name == "pytorch":
        return select_pytorch_optimizer(
            optimizer_name=optimizer_name,
            learning_rate=learning_rate,
            check_point=check_point,
            parameters=parameters,
            additional_configuration=additional_configuration,
        )
    else:
        raise Exception("Wrong library name.")


def select_tensorflow_optimizer(
    optimizer_name, learning_rate, additional_configuration
):
    if optimizer_name == "rms_prop":
        return tf.keras.optimizers.RMSprop(learning_rate=learning_rate,
                                           rho=additional_configuration.used_optim.rho,
                                           momentum=additional_configuration.used_optim.momentum,
                                          )
    elif optimizer_name == "adam":
        return tf.keras.optimizers.Adam(
            learning_rate=learning_rate,
            beta_1=additional_configuration.used_optim.beta1,
            beta_2=additional_configuration.used_optim.beta2,
            epsilon=additional_configuration.used_optim.epsilon,
        )
    elif optimizer_name == "adamax":
        return tf.keras.optimizers.Adamax(
            learning_rate=learning_rate,
            beta_1=additional_configuration.used_optim.beta1,
            beta_2=additional_configuration.used_optim.beta2,
            epsilon=additional_configuration.used_optim.epsilon,
        )
    elif optimizer_name == "adamW":
        return tfa.optimizers.AdamW(
            learning_rate=learning_rate,
            weight_decay=additional_configuration.used_optim.decay,
            beta_1=additional_configuration.used_optim.beta1,
            beta_2=additional_configuration.used_optim.beta2,
            epsilon=additional_configuration.used_optim.epsilon,
        )
    elif optimizer_name == "sgd":
        return tf.keras.optimizers.SGD(
            learning_rate=learning_rate,
            momentum=additional_configuration.used_optim.momentum,
        )
    else:
        raise Exception("No available optimizer.")


def select_pytorch_optimizer(
    optimizer_name, learning_rate, check_point, parameters, additional_configuration
):
    if check_point is None:
        if optimizer_name == "adam":
            return torch.optim.Adam(
                parameters,
                lr=learning_rate,
                betas=(
                    additional_configuration.used_optim.beta1,
                    additional_configuration.used_optim.beta2,
                ),
            )
        elif optimizer_name == "rms_prop":
            return torch.optim.RMSprop(parameters, lr=learning_rate)
        else:
            raise Exception("No available optimizer.")
    else:
        if optimizer_name == "adam":
            return torch.optim.Adam(parameters)
        elif optimizer_name == "rms_prop":
            return torch.optim.RMSprop(parameters)
        else:
            raise Exception("No available optimizer.")


#######


def select_lr_schedule(
    library_name,
    lr_scheduler_name,
    data_len,
    num_epochs,
    learning_rate,
    monitor_loss,
    name,
    optimizer,
    frequency,
    additional_configuration,
):
    if library_name == "tensorflow":
        return select_tensorflow_lr_schedule(
            lr_scheduler_name,
            data_len,
            num_epochs,
            learning_rate,
            monitor_loss,
            additional_configuration,
        )
    elif library_name == "pytorch":
        return select_pytorch_lr_schedule(
            lr_scheduler_name,
            data_len,
            num_epochs,
            learning_rate,
            monitor_loss,
            name,
            optimizer,
            frequency,
            additional_configuration,
        )
    else:
        raise Exception("Wrong library name.")


def select_tensorflow_lr_schedule(
    lr_scheduler_name,
    data_len,
    num_epochs,
    learning_rate,
    monitor_loss,
    additional_configuration,
):
    if lr_scheduler_name == "OneCycle":
        steps = data_len * num_epochs
        return tensorflow_callbacks.OneCycleScheduler(learning_rate, steps)
    elif lr_scheduler_name == "ReduceOnPlateau":
        return ReduceLROnPlateau(
            monitor=monitor_loss,
            factor=additional_configuration.used_sched.factor,
            patience=additional_configuration.used_sched.patience,
            min_lr=(learning_rate / 10),
        )
    elif lr_scheduler_name == "CosineDecay":
        decay_steps = data_len * num_epochs
        return tf.keras.optimizers.schedules.CosineDecay(
            learning_rate, decay_steps, alpha=0.0, name=None
        )
    elif lr_scheduler_name == "MultiStepScheduler":
        return tensorflow_callbacks.MultiStepScheduler(
            learning_rate,
            lr_steps=additional_configuration.used_sched.lr_steps,
            lr_rate_decay=additional_configuration.used_sched.lr_rate_decay
        )
    elif lr_scheduler_name is None:
        return None
    else:
        raise Exception("Not available Learning rate Scheduler.")


def select_pytorch_lr_schedule(
    lr_scheduler_name,
    data_len,
    num_epochs,
    learning_rate,
    monitor_loss,
    name,
    optimizer,
    frequency,
    additional_configuration,
):
    if lr_scheduler_name == "OneCycle":
        return {
            "scheduler": torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                learning_rate,
                epochs=num_epochs,
                steps_per_epoch=data_len // frequency,
            ),
            "interval": "step",
            "name": name,
            "frequency": frequency,
        }
    elif lr_scheduler_name == "ReduceOnPlateau":
        return {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=additional_configuration.used_sched.factor,
                patience=additional_configuration.used_sched.ReduceOnPlateau[
                    "patience"
                ],
                min_lr=(learning_rate / 10),
            ),
            "interval": "epoch",
            "name": name,
            "monitor": monitor_loss,
            "frequency": frequency,
        }
    else:
        raise Exception("Not available Learning rate Scheduler.")
