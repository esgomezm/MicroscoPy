import torch
print(torch.cuda.is_available())

import tensorflow as tf
print(tf.test.gpu_device_name())