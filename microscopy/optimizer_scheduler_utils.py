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
        print(f'optimizer_utils - select_tensorflow_optimizer -> Its RMSProp!')
        return tf.keras.optimizers.RMSprop(learning_rate=learning_rate,
                                           rho=additional_configuration.used_optim.rho,
                                           momentum=additional_configuration.used_optim.momentum,
                                          )
    elif optimizer_name == "adam":
        print(f'optimizer_utils - select_tensorflow_optimizer -> Its Adam!')
        return tf.keras.optimizers.Adam(
            learning_rate=learning_rate,
            beta_1=additional_configuration.used_optim.beta1,
            beta_2=additional_configuration.used_optim.beta2,
            epsilon=additional_configuration.used_optim.epsilon,
        )
    elif optimizer_name == "adamax":
        print(f'optimizer_utils - select_tensorflow_optimizer -> Its Adamax!')
        return tf.keras.optimizers.Adamax(
            learning_rate=learning_rate,
            beta_1=additional_configuration.used_optim.beta1,
            beta_2=additional_configuration.used_optim.beta2,
            epsilon=additional_configuration.used_optim.epsilon,
        )
    elif optimizer_name == "adamW":
        print(f'optimizer_utils - select_tensorflow_optimizer -> Its AdamW!')
        return tfa.optimizers.AdamW(
            learning_rate=learning_rate,
            weight_decay=additional_configuration.used_optim.decay,
            beta_1=additional_configuration.used_optim.beta1,
            beta_2=additional_configuration.used_optim.beta2,
            epsilon=additional_configuration.used_optim.epsilon,
        )
    elif optimizer_name == "sgd":
        print(f'optimizer_utils - select_tensorflow_optimizer -> Its SGD!')
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
            print(f'optimizer_utils - select_pytorch_optimizer -> Its Adam!')
            return torch.optim.Adam(
                parameters,
                lr=learning_rate,
                betas=(
                    additional_configuration.used_optim.beta1,
                    additional_configuration.used_optim.beta2,
                ),
                eps=additional_configuration.used_optim.epsilon
            )
        elif optimizer_name == "rms_prop":
            print(f'optimizer_utils - select_pytorch_optimizer -> Its RMSProp!')
            return torch.optim.RMSprop(parameters, 
                                       lr=learning_rate,
                                       betas=(
                                            additional_configuration.used_optim.beta1,
                                            additional_configuration.used_optim.beta2,
                                       ),
                                       eps=additional_configuration.used_optim.epsilon)
        elif optimizer_name == "adamax":
            print(f'optimizer_utils - select_pytorch_optimizer -> Its Adamax!')
            return torch.optim.Adamax(parameters, 
                                      lr=learning_rate,
                                      betas=(
                                        additional_configuration.used_optim.beta1,
                                        additional_configuration.used_optim.beta2,
                                      ),
                                      eps=additional_configuration.used_optim.epsilon)
        elif optimizer_name == "adamW":
            print(f'optimizer_utils - select_pytorch_optimizer -> Its AdamW!')
            return torch.optim.AdamW(parameters, 
                                     lr=learning_rate,
                                     betas=(
                                        additional_configuration.used_optim.beta1,
                                        additional_configuration.used_optim.beta2,
                                     ),
                                     eps=additional_configuration.used_optim.epsilon)
        elif optimizer_name == "sgd":
            print(f'optimizer_utils - select_pytorch_optimizer -> Its SGD!')
            return tf.keras.optimizers.SGD(
                parameters,
                learning_rate=learning_rate,
                momentum=additional_configuration.used_optim.momentum,
            )
        else:
            raise Exception("No available optimizer.")
    else:
        if optimizer_name == "adam":
            return torch.optim.Adam(parameters)
        elif optimizer_name == "rms_prop":
            return torch.optim.RMSprop(parameters)
        elif optimizer_name == "adamax":
            return torch.optim.Adamax(parameters)
        elif optimizer_name == "adamW":
            return torch.optim.AdamW(parameters)
        elif optimizer_name == "sgd":
            return tf.keras.optimizers.SGD(parameters)
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
        print(f'optimizer_utils - select_tensorflow_lr_schedule -> Its OneCycle!')
        steps = data_len * num_epochs
        return tensorflow_callbacks.OneCycleScheduler(learning_rate, steps)
    elif lr_scheduler_name == "ReduceOnPlateau":
        return ReduceLROnPlateau(
            monitor=monitor_loss,
            factor=additional_configuration.used_sched.factor,
            patience=additional_configuration.used_sched.patience,
            min_lr=(learning_rate / 10),
            mode=additional_configuration.used_sched.mode,
            verbose=1
        )
    elif lr_scheduler_name == "CosineDecay":
        print(f'optimizer_utils - select_tensorflow_lr_schedule -> Its CosineDecay!')
        decay_steps = data_len * num_epochs
        return tf.keras.optimizers.schedules.CosineDecay(
            learning_rate, decay_steps, alpha=0.0, name=None
        )
    elif lr_scheduler_name == "MultiStepScheduler":
        print(f'optimizer_utils - select_tensorflow_lr_schedule -> Its MultiStepScheduler!')
        total_steps = data_len * num_epochs
        lr_steps = [int(total_steps*i) for i in [0.5, 0.7, 0.8, 0.9]]
        return tensorflow_callbacks.MultiStepScheduler(
            learning_rate,
            lr_steps=lr_steps,
            lr_rate_decay=additional_configuration.used_sched.lr_rate_decay # 0.5
        )
    elif lr_scheduler_name is None or lr_scheduler_name == "Fixed":
        print(f'optimizer_utils - select_tensorflow_lr_schedule -> Its Fixed!')
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
        print(f'optimizer_utils - select_pytorch_lr_schedule -> Its OneCycle!')
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
        print(f'optimizer_utils - select_pytorch_lr_schedule -> Its ReduceOnPlateau!')
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
    
    elif lr_scheduler_name == "CosineDecay":
        print(f'optimizer_utils - select_pytorch_lr_schedule -> Its CosineDecay!')
        return {
            "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=data_len * num_epochs,
                eta_min=(learning_rate / 10),
            ),
            "interval": "epoch",
            "name": name,
            "monitor": monitor_loss,
            "frequency": frequency,
        } 
    elif lr_scheduler_name == "MultiStepScheduler":
        print(f'optimizer_utils - select_pytorch_lr_schedule -> Its MultiStepScheduler!')
        total_steps = num_epochs
        lr_steps = [int(total_steps*i) for i in [0.5, 0.7, 0.8, 0.9]]
        return {
            "scheduler": torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=lr_steps,
                gamma=additional_configuration.used_sched.lr_rate_decay
            ),
            "interval": "epoch",
            "name": name,
            "monitor": monitor_loss,
            "frequency": frequency,
        }
    elif lr_scheduler_name is None or lr_scheduler_name == "Fixed":
        print(f'optimizer_utils - select_pytorch_lr_schedule -> Its Fixed!')
        return {
            "scheduler": torch.optim.lr_scheduler.ConstantLR(
                optimizer,
                factor=1,
                total_iters=num_epochs,
            ),
            "interval": "epoch",
            "name": name,
            "monitor": monitor_loss,
            "frequency": frequency,
        } 
    else:
        raise Exception("Not available Learning rate Scheduler.")
