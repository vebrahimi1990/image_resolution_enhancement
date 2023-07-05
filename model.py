import tensorflow as tf
from keras.activations import sigmoid
from keras.layers import Dropout, LeakyReLU, UpSampling2D
from keras.layers.convolutional import Conv2D
from keras.layers.merge import concatenate, add, multiply
from keras.layers.pooling import GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.models import Model

regul = 'l1_l2'


def conv_block(inputs, filters, kernel):
    x = Conv2D(filters=filters, kernel_size=kernel, kernel_regularizer=regul, padding="same")(inputs)
    x = LeakyReLU(alpha=0.01)(x)
    x = Conv2D(filters=filters, kernel_size=kernel, kernel_regularizer=regul, padding="same")(x)
    y = Conv2D(filters=filters, kernel_size=1, padding="same")(inputs)
    x = add([x, y])
    x = LeakyReLU(alpha=0.01)(x)
    return x


def CAB(inputs, filters_cab, filters, kernel):
    x1 = GlobalAveragePooling2D(data_format='channels_last', keepdims=True)(inputs)
    x1 = Conv2D(filters=filters_cab, kernel_size=kernel, kernel_regularizer=regul, padding="same")(x1)
    x1 = LeakyReLU(alpha=0.01)(x1)
    x1 = Conv2D(filters=filters, kernel_size=kernel, kernel_regularizer=regul, padding="same")(x1)

    x2 = GlobalMaxPooling2D(data_format='channels_last', keepdims=True)(inputs)
    x2 = Conv2D(filters=filters_cab, kernel_size=kernel, kernel_regularizer=regul, padding="same")(x2)
    x2 = LeakyReLU(alpha=0.01)(x2)
    x2 = Conv2D(filters=filters, kernel_size=kernel, kernel_regularizer=regul, padding="same")(x2)

    z = add([x1, x2])
    z = sigmoid(z)
    z = multiply([z, inputs])
    return z


def RCAB(inputs, filters_cab, filters, kernel):
    x = Conv2D(filters=filters, kernel_size=kernel, kernel_regularizer=regul, padding="same")(inputs)
    x = LeakyReLU(alpha=0.01)(x)
    x = Conv2D(filters=filters, kernel_size=kernel, kernel_regularizer=regul, padding="same")(x)
    x = CAB(x, filters_cab=filters_cab, filters=filters, kernel=1)
    y = Conv2D(filters=filters, kernel_size=1, padding="same")(inputs)
    x = add([x, y])
    return x


def RG(inputs, num_CAB, filters, filters_cab, kernel):
    x = inputs
    for i in range(num_CAB):
        x = RCAB(x, filters_cab, filters, kernel)
    x = Conv2D(filters=filters, kernel_size=kernel, kernel_regularizer=regul, padding="same")(x)
    x = add([x, inputs])
    return x


def RiR(inputs, num_RG, num_RCAB, filters, filters_cab, kernel, dropout):
    x = inputs
    for i in range(num_RG):
        x = RG(x, num_RCAB, filters, filters_cab, kernel)
        x = Dropout(dropout)(x)
    x = Conv2D(filters=filters, kernel_size=kernel, kernel_regularizer=regul, padding="same")(x)
    x = add([x, inputs])
    return x


def make_RCAN(inputs, filters, filters_cab, num_RG, num_RCAB, kernel, dropout):
    y = Conv2D(filters=4, kernel_size=1, kernel_regularizer=regul, padding="same")(inputs)
    x = Conv2D(filters=filters, kernel_size=kernel, kernel_regularizer=regul, padding="same")(inputs)
    x = RiR(x, num_RG, num_RCAB, filters, filters_cab, kernel, dropout)
    x = Conv2D(filters=4, kernel_size=kernel, kernel_regularizer=regul, padding="same")(x)
    x = add([x, y])
    x = tf.nn.depth_to_space(x, 2)
    x = Conv2D(filters=1, kernel_size=1, padding="same")(x)
    return x


def UNet_RCAN(inputs, model_config):
    filters = sorted(model_config['filters'])
    rcan_filter = model_config['rcan_filter']
    filters_cab = model_config['filters_cab']
    num_RG = model_config['num_RG']
    num_cab = model_config['num_cab']
    kernel_shape = model_config['kernel']
    dropout = model_config['dropout']
    skip_x = []

    x = Conv2D(filters=filters[0], kernel_size=kernel_shape, padding='same')(inputs)
    x = RCAB(x, filters_cab=filters_cab, filters=filters[0], kernel=kernel_shape)
    for i, f in enumerate(filters):
        x = RCAB(x, filters_cab=filters_cab, filters=f, kernel=kernel_shape)
        x = Dropout(dropout)(x)
        skip_x.append(x)
        x = Conv2D(filters=f, kernel_size=1, strides=(2, 2), padding='same')(x)

    x = RCAB(x, filters_cab=filters_cab, filters=2 * filters[-1], kernel=kernel_shape)
    skip_x.append(x)
    x = RCAB(x, filters_cab=filters_cab, filters=2 * filters[-1], kernel=kernel_shape)
    filters.reverse()
    skip_x.reverse()

    for i, f in enumerate(filters):
        x = UpSampling2D(size=2, data_format='channels_last')(x)
        xs = skip_x[i + 1]
        x = concatenate([x, xs])
        x = RCAB(x, filters_cab=filters_cab, filters=f, kernel=kernel_shape)
        x = Dropout(dropout)(x)

    x = Conv2D(filters=1, kernel_size=1, padding="same")(x)
    x = add([x, inputs])
    y = concatenate([x, inputs])

    y = make_RCAN(inputs=y, filters=rcan_filter, filters_cab=filters_cab, num_RG=num_RG, num_RCAB=num_cab,
                  kernel=kernel_shape, dropout=dropout)
    model = Model(inputs=[inputs], outputs={'UNet': x, 'UNet_RCAN': y})
    return model
