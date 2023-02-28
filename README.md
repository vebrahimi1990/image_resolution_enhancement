# Resolution enhancement of imaging by a low NA objective lens

This repository contains a two step prediction network based on UNet and RCAN which is used for denoising and resolution enhancement of widefield microscopy images that are captured with a low numerical aperture (NA) objective lens.   
The networks are built in Tensorflow 2.7.0 framework.

# Dependencies
```
pip install -r requirements.txt
```
# Architecture


# Notebooks
```notebook_train.ipynb```
```notebook_test.ipynb```

# Training
```
git clone https://github.com/vebrahimi1990/image_resolution_enhancement.git
```

For training, specify the type of the model and add the directory to your training dataset and a directory to save the model to the configuration file ```(config.py)```.

```
python train.py
``` 


# Evaluation
For evaluation, add the directory to your test dataset and a directory to the saved model to the configuration file ```(config.py)```.
 
```
python evaluate.py
```

# Results
![plot](https://github.com/vebrahimi1990/image_resolution_enhancement/blob/master/image_files/result_cmos.png)

# Contact
Should you have any question, please contact vebrahimi1369@gmail.com. 
