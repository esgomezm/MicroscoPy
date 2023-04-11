import tensorflow as tf

from tensorflow.keras.callbacks import ReduceLROnPlateau

import torch

from . import tensorflow_callbacks

def select_optimizer(library_name, optimizer_name, learning_rate, check_point, parameters, additional_configuration):
    if library_name == 'tensorflow':
        return select_tensorflow_optimizer(optimizer_name=optimizer_name, 
                                           learning_rate=learning_rate, 
                                           additional_configuration=additional_configuration)
    elif library_name == 'pytorch':
        return select_pytorch_optimizer(optimizer_name=optimizer_name, 
                                        learning_rate=learning_rate, 
                                        check_point=check_point, 
                                        parameters=parameters, 
                                        additional_configuration=additional_configuration)
    else:
        raise Exception("Wrong library name.")
        
def select_tensorflow_optimizer(optimizer_name, learning_rate, additional_configuration):
    if optimizer_name == 'RMSprop':
        return tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    elif optimizer_name == 'Adam':
        return tf.keras.optimizers.Adam(learning_rate=learning_rate,
                                        beta_1=additional_configuration['optim']['adam']['beta1'],
                                        beta_2=additional_configuration['optim']['adam']['beta2'],
                                        epsilon=additional_configuration['optim']['adam']['epsilon'])
    elif optimizer_name == 'Adamax':
        return tf.keras.optimizers.experimental.Adamax(learning_rate=learning_rate,
				                        beta_1=additional_configuration['optim']['adamax']['beta1'],
				                        beta_2=additional_configuration['optim']['adamax']['beta2'],
				                        epsilon=additional_configuration['optim']['adamax']['epsilon'])
    elif optimizer_name == 'AdamW':
        return tf.keras.optimizers.experimental.AdamW(learning_rate=learning_rate,
				                        weight_decay=additional_configuration['optim']['adamW']['weight_decay'],
				                        beta_1=additional_configuration['optim']['adamW']['beta1'],
				                        beta_2=additional_configuration['optim']['adamW']['beta2'],
				                        epsilon=additional_configuration['optim']['adamW']['epsilon'])
    elif optimizer_name == 'SGD':
        return tf.keras.optimizers.SGD(learning_rate=learning_rate,
        				momentum=additional_configuration['optim']['sgd_momentum'])
    else:
        raise Exception("No available optimizer.")

def select_pytorch_optimizer(optimizer_name, learning_rate, check_point, parameters, additional_configuration):
    
    print(additional_configuration)

    if check_point is None:
        if optimizer_name == 'Adam':
            return torch.optim.Adam(parameters, lr=learning_rate, betas=(additional_configuration['optim']['adam']['beta1'],
                                                                         additional_configuration['optim']['adam']['beta2']))
        elif optimizer_name == 'RMSprop':
            return torch.optim.RMSprop(parameters, lr=learning_rate)
        else:
            raise Exception("No available optimizer.")
    else:
        if optimizer_name == 'Adam':
            return torch.optim.Adam(parameters)
        elif optimizer_name == 'RMSprop':
            return torch.optim.RMSprop(parameters)
        else:
            raise Exception("No available optimizer.")

#######

def select_lr_schedule(library_name, lr_scheduler_name, data_len, number_of_epochs, learning_rate, 
                       monitor_loss, name, optimizer, frequency, additional_configuration):
    if library_name == 'tensorflow':
        return select_tensorflow_lr_schedule(lr_scheduler_name, data_len, number_of_epochs, learning_rate, 
                                             monitor_loss, additional_configuration)
    elif library_name == 'pytorch':
        return select_pytorch_lr_schedule(lr_scheduler_name, data_len, number_of_epochs, learning_rate, 
                                          monitor_loss, name, optimizer, frequency, additional_configuration)
    else:
        raise Exception("Wrong library name.")

def select_tensorflow_lr_schedule(lr_scheduler_name, data_len, number_of_epochs, 
                                  learning_rate, monitor_loss, additional_configuration):
    if lr_scheduler_name == 'OneCycle':
        steps = data_len * number_of_epochs
        return tensorflow_callbacks.OneCycleScheduler(learning_rate, steps)
    elif lr_scheduler_name == 'ReduceOnPlateau':
        return ReduceLROnPlateau(monitor=monitor_loss,
        			factor=additional_configuration['optim']['ReduceOnPlateau']['factor'], 
        			patience=additional_configuration['optim']['ReduceOnPlateau']['patience'], 
                    min_lr=(learning_rate/10))
    elif lr_scheduler_name == 'CosineDecay':
        decay_steps = data_len * number_of_epochs
        return tf.keras.optimizers.schedules.CosineDecay(learning_rate, decay_steps, alpha=0.0, name=None)
    elif lr_scheduler_name == 'MultiStepScheduler':
        return tensorflow_callbacks.MultiStepScheduler(learning_rate,
        			  lr_steps=additional_configuration['optim']['MultiStepScheduler']['lr_steps'], 
        			  lr_rate_decay=additional_configuration['optim']['MultiStepScheduler']['lr_rate_decay'])
    elif lr_scheduler_name is None:
        return None
    else:
        raise Exception("Not available Learning rate Scheduler.")  

def select_pytorch_lr_schedule(lr_scheduler_name, data_len, number_of_epochs, 
                               learning_rate, monitor_loss, name, optimizer, frequency, additional_configuration):
    
    if lr_scheduler_name == 'OneCycle':
        return {'scheduler': torch.optim.lr_scheduler.OneCycleLR(
                optimizer, 
				learning_rate, 
				epochs=number_of_epochs, 
				steps_per_epoch=data_len//frequency
				),
                'interval': 'step',
                'name': name,
                'frequency': frequency
				}
    elif lr_scheduler_name == 'ReduceOnPlateau':
        return {'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=additional_configuration['optim']['ReduceOnPlateau']['factor'],
                patience=additional_configuration['optim']['ReduceOnPlateau']['patience'],
                min_lr=(learning_rate/10)
                ),
                'interval': 'epoch',
                'name': name,
                'monitor': monitor_loss,
                'frequency': frequency
				}
    else:
        raise Exception("Not available Learning rate Scheduler.")  