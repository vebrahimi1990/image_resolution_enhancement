import os
import numpy as np
import tensorflow as tf
from keras.models import Input
from tifffile import imwrite
from config import CFG
from datagenerator import data_generator, data_generator_test
from model import UNet_RCAN

gpus = tf.config.list_physical_devices('GPU')
tf.config.set_logical_device_configuration(gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=12000)])
print(len(gpus), "Physical GPUs")

data_config = CFG['data_test']
model_config = CFG['model']
if data_config['train']:
    x_test, w_test, y_test = data_generator(data_config)

else:
    x_test = data_generator_test(data_config)


patch_size = int(data_config['patch_size']*data_config['scale'])
model_input = Input((data_config['patch_size'], data_config['patch_size'], 1))
model = eval(model_config['model_type'] + "(model_input, model_config)")
model(np.zeros((1, data_config['patch_size'], data_config['patch_size'], 1)))
model.load_weights(model_config['save_dr'])

prediction1 = np.zeros((len(x_test), data_config['patch_size'], data_config['patch_size'], 1))
prediction2 = np.zeros((len(x_test), patch_size, patch_size, 1))

for i in range(len(x_test)):
    prediction = model(x_test[i:i + 1], training=False)
    prediction1[i] = prediction['UNet']
    prediction2[i] = prediction[model_config['model_type']]
    prediction1[i] = prediction1[i] / prediction1[i].max()
    prediction2[i] = prediction2[i] / prediction2[i].max()
prediction1[prediction1 < 0] = 0
prediction2[prediction2 < 0] = 0

if not data_config['train']:
    pred1 = (prediction1 * (2 ** 16 - 1)).astype(np.uint16)
    pred2 = (prediction2 * (2 ** 16 - 1)).astype(np.uint16)
    X_test = (x_test * (2 ** 16 - 1)).astype(np.uint16)
    imwrite(os.path.join(data_config['save_dr'], '', 'pred_scaled.tif'), pred1.squeeze(), imagej=True,
            metadata={'axes': 'TYX'})
    imwrite(os.path.join(data_config['save_dr'], '', 'pred.tif'), pred2.squeeze(), imagej=True,
            metadata={'axes': 'TYX'})
    imwrite(os.path.join(data_config['save_dr'], '', 'noisy.tif'), X_test.squeeze(), imagej=True,
            metadata={'axes': 'TYX'})
if data_config['train']:
    Y_test = (y_test * (2 ** 16 - 1)).astype(np.uint16)
    imwrite(os.path.join(data_config['save_dr'], '', 'gt.tif'), Y_test.squeeze(), imagej=True, metadata={'axes': 'TYX'})
