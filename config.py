CFG = {
    "data_train": {
        "deconv_image_dr": r"D:\Projects\MSD_UNet_RCAN\widefield\deconvolution\histon-tubulin\gt",
        "wf_image_dr": r"D:\Projects\MSD_UNet_RCAN\widefield\deconvolution\histon-tubulin\gt",
        "patch_size": 128,
        "n_patches": 8,
        "n_channel": 0,
        "threshold": 0.4,
        "fr_start": 0,
        "fr_end": 11,
        "projection": "MIP",
        "scale": 2,
        "psf_filter": 1.5,
        "lp": 0.5,
        "add_noise": True,
        "shuffle": True,
        "augment": False,
        "train": True
    },
    "data_test": {
        "deconv_image_dr": r"D:\Projects\MSD_UNet_RCAN\widefield\deconvolution\histon-tubulin\gt",
        "wf_image_dr": r"D:\Projects\MSD_UNet_RCAN\widefield\deconvolution\histon-tubulin\gt",
        "lowNA_image_dr": r"D:\Projects\MSD_UNet_RCAN\40x-NA0.75\low NA\ddx39b\0.tif",
        "save_dr": r"D:\Projects\Denoising-STED",
        "patch_size": 256,
        "n_patches": 2,
        "n_channel": 0,
        "threshold": 0.0,
        "fr_start": 0,
        "fr_end": 11,
        "projection": "MIP",
        "scale": 2,
        "psf_filter": 1.5,
        "lp": 0.5,
        "add_noise": False,
        "shuffle": False,
        "augment": False,
        "train": False
    },
    "model": {
        "model_type": 'UNet_RCAN',
        "filters": [64, 128, 256],
        "filters_cab": 4,
        "num_RG": 3,
        "num_cab": 8,
        "kernel": 3,
        "dropout": 0.2,
        "lr": 0.0001,
        "n_epochs": 200,
        "batch_size": 1,
        "save_dr": r"D:\Projects\Denoising-STED\model.h5",
        "save_config": r"D:\Projects\Denoising-STED"
    },
    "callbacks": {
        "patience_stop": 20,
        "factor_lr": 0.2,
        "patience_lr": 5,
        "num_cab": 8,
        "kernel": 3,
        "dropout": 0.2,
        "lr": 0.0001,
        "n_epochs": 200,
        "batch_size": 1,
        "save_dr": r"D:\Projects\Denoising-STED\model.h5"
    }
}
