import random
import glob
import numpy as np
from tifffile import imread
from scipy.ndimage import gaussian_filter
import cv2


# Loads 3D image stacks (tif files) and normalize them
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


# Generates training data by downscaling and projecting high-resolution 3D image stacks
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

    # Load the deconvolved and widefield 3D stacks
    gt, wf = stack_generator_3D(GT_dr, low_dr, fr_start, fr_end)

    # Generate maximum/average intensity projection stacks as ground-truths
    if data_config['projection'] == "AIP":
        gt = np.mean(gt, axis=-2)
        wf = np.mean(wf, axis=-2)
    if data_config['projection'] == "MIP":
        gt = np.max(gt, axis=-2)
        wf = np.max(wf, axis=-2)

    print(gt.shape)

    # Patch size after upscaling
    scale = int(patch_size * scale)

    # Noisy patches
    x = np.empty((gt.shape[0] * n_patches * n_patches, patch_size, patch_size, 1), dtype=np.float64)
    # High SNR patches
    w = np.empty((gt.shape[0] * n_patches * n_patches, patch_size, patch_size, 1), dtype=np.float64)
    # High SNR & high-resolution patches
    y = np.empty((gt.shape[0] * n_patches * n_patches, scale, scale, 1), dtype=np.float64)

    # Generate the coordinates of patches to be generated from projected images
    rr = np.floor(np.linspace(0, gt.shape[2] - scale, n_patches)).astype(np.int32)
    cc = np.floor(np.linspace(0, gt.shape[2] - scale, n_patches)).astype(np.int32)

    count = 0
    for m in range(gt.shape[0]):
        for j in range(n_patches):
            for k in range(n_patches):
                # The kernel size of PSF to degrade the resolution of GT data (25% variation)
                psf_filter_rand = psf_filter + psf_filter * (np.random.rand() - 0.5) / 4
                y[count, :, :, 0] = gt[m, rr[j]:rr[j] + scale, cc[k]:cc[k] + scale, n_channel]

                # Gaussian filtering to degrade the image resolution
                wf_filtered = gaussian_filter(
                    wf[m, rr[j]:rr[j] + scale, cc[k]:cc[k] + scale, n_channel], sigma=psf_filter_rand)

                # Downscaling the filtered images
                w[count, :, :, 0] = cv2.resize(wf_filtered, dsize=(patch_size, patch_size),
                                               interpolation=cv2.INTER_CUBIC)

                # w[count, :, :, 0] = wf_filtered # Uncomment if the input and output pixel sizes are the same

                x[count, :, :, 0] = w[count, :, :, 0]
                count = count + 1

    # Adding poisson noise to low resolution images to generate noisy inputs
    x = generate_noisy_image(x, lp, add_noise=add_noise)

    # Data augmentation by flipping and rotating the images
    xx, ww, yy = data_augmentation(x, w, y, augment=augment)

    # Select patches that their normalized average intensity is higher than a threshold
    xxx, www, yyy = patch_selector(xx, ww, yy, threshold)

    # Shuffle the image stacks
    xxs, wws, yys = data_shuffler(xxx, www, yyy, shuffle=shuffle)

    # Normalize the image stacks
    x_train = xxs / (xxs.max(axis=(1, 2))).reshape((xxs.shape[0], 1, 1, 1))
    w_train = wws / (wws.max(axis=(1, 2))).reshape((wws.shape[0], 1, 1, 1))
    y_train = yys / (yys.max(axis=(1, 2))).reshape((yys.shape[0], 1, 1, 1))

    print('Dataset shape:', x_train.shape)
    return x_train, w_train, y_train


def generate_noisy_image(input, lp, add_noise=False):
    x = np.zeros(input.shape)
    if add_noise:
        for i in range(len(input)):
            lambp = lp + lp * (np.random.rand() - 0.5)
            x[i] = np.random.poisson(input[i] * lambp, size=input[i].shape)
    else:
        x = input
    x = x / (x.max(axis=(1, 2))).reshape((x.shape[0], 1, 1, 1))
    return x


def data_augmentation(input, output1, output2, augment=False):
    if augment:
        count = input.shape[0]
        x = np.zeros(((4 * count,) + input.shape[1:]), dtype=np.float64)
        w = np.zeros(((4 * count,) + input.shape[1:]), dtype=np.float64)
        y = np.zeros(((4 * count,) + output2.shape[1:]), dtype=np.float64)

        x[0:count, :, :, :] = input
        x[count:2 * count, :, :, :] = np.flip(input, axis=1)
        x[2 * count:3 * count, :, :, :] = np.flip(input, axis=2)
        x[3 * count:4 * count, :, :, :] = np.flip(input, axis=(1, 2))

        w[0:count, :, :, :] = output1
        w[count:2 * count, :, :, :] = np.flip(output1, axis=1)
        w[2 * count:3 * count, :, :, :] = np.flip(output1, axis=2)
        w[3 * count:4 * count, :, :, :] = np.flip(output1, axis=(1, 2))

        y[0:count, :, :, :] = output2
        y[count:2 * count, :, :, :] = np.flip(output2, axis=1)
        y[2 * count:3 * count, :, :, :] = np.flip(output2, axis=2)
        y[3 * count:4 * count, :, :, :] = np.flip(output2, axis=(1, 2))
    else:
        x = input
        w = output1
        y = output2

    return x, w, y


def patch_selector(input, output1, output2, threshold):
    norm = np.linalg.norm(output2, axis=(1, 2))
    norm = norm / norm.max()
    ind_norm = np.where(norm > threshold)[0]

    x = np.empty((len(ind_norm),) + input.shape[1:])
    w = np.empty((len(ind_norm),) + output1.shape[1:])
    y = np.empty((len(ind_norm),) + output2.shape[1:])

    for i in range(len(ind_norm)):
        x[i] = x[ind_norm[i]]
        w[i] = w[ind_norm[i]]
        y[i] = y[ind_norm[i]]
    return x, w, y


def data_shuffler(input, output1, output2, shuffle=False):
    if shuffle:
        ind_shuffled = np.linspace(0, len(input) - 1, len(input))
        random.shuffle(ind_shuffled)
        ind_shuffled = ind_shuffled.astype(int)

        x = np.empty(input.shape, dtype=np.float64)
        w = np.empty(output1.shape, dtype=np.float64)
        y = np.empty(output2.shape, dtype=np.float64)
        for i in range(len(input)):
            x[i] = input[ind_shuffled[i]]
            w[i] = output1[ind_shuffled[i]]
            y[i] = output2[ind_shuffled[i]]
    else:
        x = input
        w = output1
        y = output2
    return x, w, y


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
    low = low / low.max(axis=(-1, -2)).reshape(low.shape[0:2] + (1, 1))

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
