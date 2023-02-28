from keras.activations import sigmoid
from keras.layers import Dropout, LeakyReLU, UpSampling2D
from keras.layers.convolutional import Conv2D
from keras.layers.merge import concatenate, add, multiply
from keras.layers.pooling import MaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
import tensorflow as tf


def conv_block(inputs, filters, kernel):
    x = Conv2D(filters=filters, kernel_size=kernel, padding="same")(inputs)
    x = LeakyReLU()(x)
    x = Conv2D(filters=filters, kernel_size=kernel, padding="same")(x)
    y = Conv2D(filters=filters, kernel_size=3, padding="same")(inputs)
    x = add([x, y])
    x = LeakyReLU()(x)
    return x


def CAB(inputs, filters_cab, filters, kernel):
    x = Conv2D(filters=filters, kernel_size=kernel, padding="same")(inputs)
    x = LeakyReLU()(x)
    x = Conv2D(filters=filters, kernel_size=kernel, padding="same")(x)
    z = GlobalAveragePooling2D(data_format='channels_last', keepdims=True)(x)
    z = Conv2D(filters=filters_cab, kernel_size=1, padding="same")(z)
    z = LeakyReLU()(z)
    z = Conv2D(filters=filters, kernel_size=1, padding="same")(z)
    z = sigmoid(z)
    z = multiply([z, x])
    z = add([z, inputs])
    return z


def RG(inputs, num_CAB, filters, filters_cab, kernel):
    x = inputs
    for i in range(num_CAB):
        x = CAB(x, filters_cab, filters, kernel)
    x = Conv2D(filters=filters, kernel_size=kernel, padding="same")(x)
    x = add([x, inputs])
    return x


def RiR(inputs, num_RG, num_RCAB, filters, filters_cab, kernel, dropout):
    x = inputs
    for i in range(num_RG):
        x = RG(x, num_RCAB, filters, filters_cab, kernel)
        x = Dropout(dropout)(x)
    x = Conv2D(filters=filters, kernel_size=kernel, padding="same")(x)
    x = add([x, inputs])
    return x


def make_RCAN(inputs, filters, filters_cab, num_RG, num_RCAB, kernel, dropout):
    x = Conv2D(filters=filters, kernel_size=kernel, padding="same")(inputs)
    x = RiR(x, num_RG, num_RCAB, filters, filters_cab, kernel, dropout)
    x = Conv2D(filters=4, kernel_size=3, padding="same")(x)
    x = tf.nn.depth_to_space(x, 2)
    x = Conv2D(filters=1, kernel_size=1, padding="same")(x)
    return x


def UNet_RCAN(inputs, model_config):
    filters = sorted(model_config['filters'])
    filters_cab = model_config['filters_cab']
    num_RG = model_config['num_RG']
    num_cab = model_config['num_cab']
    kernel_shape = model_config['kernel']
    dropout = model_config['dropout']
    skip_x = []
    x = inputs
    for i, f in enumerate(filters):
        x = conv_block(x, f, kernel_shape)
        x = Dropout(dropout)(x)
        skip_x.append(x)
        x = MaxPooling2D(2)(x)

    x = conv_block(x, 2 * filters[-1], kernel_shape)
    skip_x.append(x)
    x = conv_block(x, 2 * filters[-1], kernel_shape)
    filters.reverse()
    skip_x.reverse()

    for i, f in enumerate(filters):
        x = UpSampling2D(size=2, data_format='channels_last')(x)
        xs = skip_x[i + 1]
        xs = CAB(xs, filters_cab=filters_cab, filters=f, kernel=kernel_shape)
        x = concatenate([x, xs])
        x = conv_block(x, f, kernel_shape)
        x = Dropout(dropout)(x)

    x = Conv2D(filters=1, kernel_size=1, padding="same")(x)
    y = concatenate([x, inputs])

    y = make_RCAN(inputs=y, filters=filters[0], filters_cab=filters_cab, num_RG=num_RG, num_RCAB=num_cab,
                  kernel=kernel_shape, dropout=dropout)
    model = Model(inputs=[inputs], outputs={'UNet': x, 'UNet_RCAN': y})
    return model



