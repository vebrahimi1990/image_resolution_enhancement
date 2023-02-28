import os
import tensorflow as tf
from tensorflow import keras
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.models import Input
from config import CFG
from datagenerator import data_generator
from loss import loss
from model import UNet_RCAN

gpus = tf.config.list_physical_devices('GPU')
tf.config.set_logical_device_configuration(gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=12000)])
print(len(gpus), "Physical GPUs")

data_config = CFG['data_train']
model_config = CFG['model']
callback = CFG['callbacks']
x_train, w_train, y_train = data_generator(data_config)

model_input = Input((data_config['patch_size'], data_config['patch_size'], 1))
model = eval(model_config['model_type'] + "(model_input, model_config)")
# print(model.summary())

optimizer = keras.optimizers.Adam(learning_rate=model_config['lr'])
model.compile(optimizer=optimizer, loss={'UNet': loss, 'UNet_RCAN': loss})

callbacks = [
    EarlyStopping(patience=callback['patience_stop'], verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=callback['factor_lr'], patience=callback['patience_lr']),
    ModelCheckpoint(filepath=model_config['save_dr'], verbose=1, save_best_only=True, save_weights_only=True)]

with open(os.path.join(model_config['save_config'], '', 'configuration.txt'), 'w') as data:
    data.write(str(CFG['model']))

results = model.fit(x=x_train, y={'UNet': w_train, 'UNet_RCAN': y_train}, batch_size=model_config['batch_size'],
                    epochs=model_config['n_epochs'],
                    verbose=1, callbacks=callbacks, validation_split=0.1)
