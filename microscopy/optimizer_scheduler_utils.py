import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.callbacks import ReduceLROnPlateau

import torch

from . import custom_callbacks


#####################################
#
# Functions to select an optimizer.

def select_optimizer(
    library_name,
    optimizer_name,
    learning_rate,
    check_point,
    parameters,
    additional_configuration,
):
    """
    Selects the appropriate optimizer based on the given library name and returns it.

    Parameters:
        library_name (str): The name of the library to select optimizer from.
        optimizer_name (str): The name of the optimizer to select.
        learning_rate (float): The learning rate for the optimizer.
        check_point (str): The checkpoint for pytorch optimizer (required only for pytorch).
        parameters (list): The parameters for pytorch optimizer (required only for pytorch).
        additional_configuration (dict): Additional configuration for the optimizer.

    Returns:
        The selected optimizer.

    Raises:
        Exception: If an invalid library name is provided.
    """
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
    """
    Selects and returns a TensorFlow optimizer based on the provided optimizer name,
    learning rate, and additional configuration.

    Parameters:
        optimizer_name (str): The name of the optimizer to select.
        learning_rate (float): The learning rate for the optimizer.
        additional_configuration (object): An object containing additional configuration parameters.

    Returns:
        tf.keras.optimizers.Optimizer: The selected TensorFlow optimizer.

    Raises:
        Exception: If no available optimizer matches the provided optimizer name.
    """
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
    """
    Selects and initializes a PyTorch optimizer based on the given parameters.

    Parameters:
        optimizer_name (str): The name of the optimizer to be selected.
        learning_rate (float): The learning rate to be used by the optimizer.
        check_point (Any): A checkpoint to determine if additional configuration is needed.
        parameters (Iterable): The parameters to be optimized.
        additional_configuration (Any): Additional configuration parameters.

    Returns:
        torch.optim.Optimizer: The initialized PyTorch optimizer.

    Raises:
        Exception: If no available optimizer is found.
    """
    if check_point is None:
        if optimizer_name == "adam":
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
            return torch.optim.RMSprop(parameters, 
                                       lr=learning_rate,
                                       betas=(
                                            additional_configuration.used_optim.beta1,
                                            additional_configuration.used_optim.beta2,
                                       ),
                                       eps=additional_configuration.used_optim.epsilon)
        elif optimizer_name == "adamax":
            return torch.optim.Adamax(parameters, 
                                      lr=learning_rate,
                                      betas=(
                                        additional_configuration.used_optim.beta1,
                                        additional_configuration.used_optim.beta2,
                                      ),
                                      eps=additional_configuration.used_optim.epsilon)
        elif optimizer_name == "adamW":
            return torch.optim.AdamW(parameters, 
                                     lr=learning_rate,
                                     betas=(
                                        additional_configuration.used_optim.beta1,
                                        additional_configuration.used_optim.beta2,
                                     ),
                                     eps=additional_configuration.used_optim.epsilon)
        elif optimizer_name == "sgd":
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

#
#####################################

#####################################
#
# Functions to select a learning rate scheduler.

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
    """
    Selects the appropriate learning rate schedule based on the given library name.

    Parameters:
        library_name (str): The name of the deep learning library ("tensorflow" or "pytorch").
        lr_scheduler_name (str): The name of the learning rate scheduler.
        data_len (int): The length of the dataset.
        num_epochs (int): The number of epochs.
        learning_rate (float): The initial learning rate.
        monitor_loss (bool): Whether to monitor the loss.
        name (str): The name of the model.
        optimizer (str): The name of the optimizer.
        frequency (int): The frequency of the learning rate update.
        additional_configuration (dict): Additional configuration parameters.

    Returns:
        The selected learning rate schedule.

    Raises:
        Exception: If the library name is not "tensorflow" or "pytorch".
    """
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
    """
    Generates a TensorFlow learning rate schedule based on the given parameters.

    Args:
        lr_scheduler_name (str): The name of the learning rate scheduler.
        data_len (int): The length of the data.
        num_epochs (int): The number of epochs.
        learning_rate (float): The initial learning rate.
        monitor_loss (str): The loss to monitor.
        additional_configuration (object): Additional configuration for the scheduler.

    Returns:
        tf.keras.optimizers.schedules.LearningRateSchedule or None: The generated learning rate schedule.

    Raises:
        Exception: If the specified learning rate scheduler is not available.
    """
    if lr_scheduler_name == "OneCycle":
        steps = data_len * num_epochs
        return custom_callbacks.OneCycleScheduler(learning_rate, steps)
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
        decay_steps = data_len * num_epochs
        return tf.keras.optimizers.schedules.CosineDecay(
            learning_rate, decay_steps, alpha=0.0, name=None
        )
    elif lr_scheduler_name == "MultiStepScheduler":
        total_steps = data_len * num_epochs
        lr_steps = [int(total_steps*i) for i in [0.5, 0.7, 0.8, 0.9]]
        return custom_callbacks.MultiStepScheduler(
            learning_rate,
            lr_steps=lr_steps,
            lr_rate_decay=additional_configuration.used_sched.lr_rate_decay # 0.5
        )
    elif lr_scheduler_name is None or lr_scheduler_name == "Fixed":
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
    """
    Selects and returns a PyTorch learning rate schedule based on the provided parameters.

    Parameters:
        - lr_scheduler_name (str): The name of the learning rate scheduler to select.
        - data_len (int): The length of the training data.
        - num_epochs (int): The number of epochs to train for.
        - learning_rate (float): The initial learning rate.
        - monitor_loss (str): The name of the loss function to monitor.
        - name (str): The name of the learning rate schedule.
        - optimizer (torch.optim.Optimizer): The optimizer object.
        - frequency (int): The frequency of the learning rate schedule.
        - additional_configuration (Any): Additional configuration parameters.

    Returns:
        dict: A dictionary containing the selected learning rate scheduler, interval, name, monitor, and frequency.

    Raises:
        Exception: If the provided learning rate scheduler is not available.
    """
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
    
    elif lr_scheduler_name == "CosineDecay":
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

#
#####################################
