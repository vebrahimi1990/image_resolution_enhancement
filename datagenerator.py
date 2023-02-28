import random
import glob
import numpy as np
from tifffile import imread
from scipy.ndimage import gaussian_filter
import cv2


def stack_generator_3D(GT_dr, low_dr, fr_start, fr_end):
    path_gt = GT_dr + '/*.tif'
    path_low = low_dr + '/*.tif'
    image_gt = imread(sorted(glob.glob(path_gt))).astype(np.float32)
    image_low = imread(sorted(glob.glob(path_low))).astype(np.float32)

    if len(image_gt.shape) == 3:
        image_gt = np.reshape(image_gt, (image_gt.shape[0], 1, 1, image_gt.shape[1], image_gt.shape[2]))
        image_low = np.reshape(image_low,
                               (image_low.shape[0], 1, 1, image_low.shape[1], image_low.shape[2]))

    if len(image_gt.shape) == 4:
        image_gt = np.reshape(image_gt, (image_gt.shape[0], image_gt.shape[1], 1, image_gt.shape[2], image_gt.shape[3]))
        image_low = np.reshape(image_low,
                               (image_low.shape[0], image_low.shape[1], 1, image_low.shape[2], image_low.shape[3]))

    print(image_gt.shape)
    for i in range(len(image_gt)):
        for j in range(image_gt.shape[2]):
            if image_gt[i, :, j, :, :].max() > 0:
                image_gt[i, :, j, :, :] = image_gt[i, :, j, :, :] / image_gt[i, :, j, :, :].max()
            if image_low[i, :, j, :, :].max() > 0:
                image_low[i, :, j, :, :] = image_low[i, :, j, :, :] / image_low[i, :, j, :, :].max()

    crop_gt = image_gt[:, fr_start:fr_end, :, :, :]
    crop_low = image_low[:, fr_start:fr_end, :, :, :]
    crop_gt = np.moveaxis(crop_gt, 1, -1)
    crop_low = np.moveaxis(crop_low, 1, -1)
    crop_gt = np.moveaxis(crop_gt, 1, -1)
    crop_low = np.moveaxis(crop_low, 1, -1)
    print(crop_low.shape)
    return crop_gt, crop_low


def data_generator(data_config):
    GT_dr = data_config['deconv_image_dr']
    low_dr = data_config['wf_image_dr']
    patch_size = data_config['patch_size']
    n_patches = data_config['n_patches']
    n_channel = data_config['n_channel']
    threshold = data_config['threshold']
    fr_start = data_config['fr_start']
    fr_end = data_config['fr_end']
    scale = data_config['scale']
    psf_filter = data_config['psf_filter']
    lp = data_config['lp']
    augment = data_config['augment']
    shuffle = data_config['shuffle']
    add_noise = data_config['add_noise']

    gt, wf = stack_generator_3D(GT_dr, low_dr, fr_start, fr_end)

    scale = int(patch_size*scale)

    if data_config['projection'] == "AIP":
        gt = np.mean(gt, axis=-2)
        wf = np.mean(wf, axis=-2)
    if data_config['projection'] == "MIP":
        gt = np.max(gt, axis=-2)
        wf = np.max(wf, axis=-2)

    print(gt.shape)
    m = gt.shape[0]
    img_size = gt.shape[2]

    x = np.empty((m * n_patches * n_patches, patch_size, patch_size, 1), dtype=np.float64)
    w = np.empty((m * n_patches * n_patches, patch_size, patch_size, 1), dtype=np.float64)
    y = np.empty((m * n_patches * n_patches, scale, scale, 1), dtype=np.float64)

    rr = np.floor(np.linspace(0, img_size - scale, n_patches)).astype(np.int32)
    cc = np.floor(np.linspace(0, img_size - scale, n_patches)).astype(np.int32)

    count = 0
    for l in range(m):
        for j in range(n_patches):
            for k in range(n_patches):
                y[count, :, :, 0] = gt[l, rr[j]:rr[j] + scale, cc[k]:cc[k] + scale, n_channel]
                wf_filtered = gaussian_filter(
                    wf[l, rr[j]:rr[j] + patch_size, cc[k]:cc[k] + scale, n_channel], sigma=psf_filter)
                x[count, :, :, 0] = cv2.resize(wf_filtered, dsize=(patch_size, patch_size),
                                               interpolation=cv2.INTER_CUBIC)
                w[count, :, :, 0] = cv2.resize(wf_filtered, dsize=(patch_size, patch_size),
                                               interpolation=cv2.INTER_CUBIC)
                count = count + 1

    if add_noise:
        for i in range(len(x)):
            x[i] = np.random.poisson(x[i] / lp, size=x[i].shape)
    x = x / (x.max(axis=(1, 2))).reshape((x.shape[0], 1, 1, 1))

    if augment:
        count = x.shape[0]
        xx = np.zeros((4 * count, patch_size, patch_size, 1), dtype=np.float64)
        ww = np.zeros((4 * count, patch_size, patch_size, 1), dtype=np.float64)
        yy = np.zeros((4 * count, scale, scale, 1), dtype=np.float64)

        xx[0:count, :, :, :] = x
        xx[count:2 * count, :, :, :] = np.flip(x, axis=1)
        xx[2 * count:3 * count, :, :, :] = np.flip(x, axis=2)
        xx[3 * count:4 * count, :, :, :] = np.flip(x, axis=(1, 2))

        ww[0:count, :, :, :] = w
        ww[count:2 * count, :, :, :] = np.flip(w, axis=1)
        ww[2 * count:3 * count, :, :, :] = np.flip(w, axis=2)
        ww[3 * count:4 * count, :, :, :] = np.flip(w, axis=(1, 2))

        yy[0:count, :, :, :] = y
        yy[count:2 * count, :, :, :] = np.flip(y, axis=1)
        yy[2 * count:3 * count, :, :, :] = np.flip(y, axis=2)
        yy[3 * count:4 * count, :, :, :] = np.flip(y, axis=(1, 2))
    else:
        xx = x
        ww = w
        yy = y

    norm_x = np.linalg.norm(yy, axis=(1, 2))
    norm_x = norm_x / norm_x.max()
    ind_norm = np.where(norm_x > threshold)[0]

    xxx = np.empty((len(ind_norm), xx.shape[1], xx.shape[2], xx.shape[3]))
    www = np.empty((len(ind_norm), ww.shape[1], ww.shape[2], ww.shape[3]))
    yyy = np.empty((len(ind_norm), yy.shape[1], yy.shape[2], yy.shape[3]))

    for i in range(len(ind_norm)):
        xxx[i] = xx[ind_norm[i]]
        www[i] = ww[ind_norm[i]]
        yyy[i] = yy[ind_norm[i]]

    aa = np.linspace(0, len(xxx) - 1, len(xxx))
    random.shuffle(aa)
    aa = aa.astype(int)

    xxs = np.empty(xxx.shape, dtype=np.float64)
    wws = np.empty(www.shape, dtype=np.float64)
    yys = np.empty(yyy.shape, dtype=np.float64)

    if shuffle:
        for i in range(len(xxx)):
            xxs[i] = xxx[aa[i]]
            wws[i] = www[aa[i]]
            yys[i] = yyy[aa[i]]
    else:
        xxs = xxx
        wws = www
        yys = yyy

    xxs[xxs < 0] = 0
    for i in range(len(xxs)):
        for j in range(n_channel):
            xxs[i, :, :, j] = xxs[i, :, :, j] / xxs[i, :, :, j].max()
            wws[i, :, :, j] = wws[i, :, :, j] / wws[i, :, :, j].max()
            if yys[i, :, :, j].max() > 0:
                yys[i, :, :, j] = yys[i, :, :, j] / yys[i, :, :, j].max()

    x_train = xxs
    w_train = wws
    y_train = yys

    print('Dataset shape:', x_train.shape)
    return x_train, w_train, y_train


def data_generator_test(data_config):
    low_dr = data_config['lowNA_image_dr']
    patch_size = data_config['patch_size']
    n_patches = data_config['n_patches']
    n_channel = data_config['n_channel']
    lp = data_config['lp']
    add_noise = data_config['add_noise']

    low = imread(low_dr).astype(np.float64)
    if len(low.shape) == 3:
        low = np.reshape(low, (low.shape[0], 1, low.shape[1], low.shape[2]))
    if len(low.shape) == 2:
        low = np.reshape(low, (1, 1, low.shape[0], low.shape[1]))
    print(low.shape)
    m = low.shape[0]
    img_size = low.shape[2]

    low = low / (low.max(axis=(-1, -2))).reshape((low.shape[0], low.shape[1], 1, 1))

    if add_noise:
        for i in range(len(low)):
            low[i] = np.random.poisson(low[i] / lp, size=low[i].shape)

    x = np.empty((m * n_patches * n_patches, patch_size, patch_size, 1), dtype=np.float64)

    rr = np.floor(np.linspace(0, img_size - patch_size, n_patches)).astype(np.int32)
    cc = np.floor(np.linspace(0, low.shape[3] - patch_size, n_patches)).astype(np.int32)

    count = 0
    for l in range(m):
        for j in range(n_patches):
            for k in range(n_patches):
                x[count, :, :, 0] = low[l, n_channel, rr[j]:rr[j] + patch_size, cc[k]:cc[k] + patch_size]
                count = count + 1

    for i in range(len(x)):
        for j in range(n_channel):
            x[i, :, :, j] = x[i, :, :, j] / x[i, :, :, j].max()

    x_train = x

    print('Dataset shape:', x_train.shape)
    return x_train
