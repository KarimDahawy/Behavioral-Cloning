# **Behavioral Cloning**


## Overview
-------------------

In this project we will use  deep neural networks and convolutional neural networks to clone driving behavior.

We will train a model which will be able to predict  a steering angle of an autonomous vehicle.

After that we will use this model to drive the vehicle autonomously around a track in a simulator provided by Udacity team.

## Behavioral Cloning Project Goals
-------------------------------------------------------------

**The goals / steps of this project are the following:**

	1. Use the simulator to collect data of good driving behavior.
	2. Design, train and validate a model that predicts a steering angle from image data.
	3. Use the model to drive the vehicle autonomously around the first track in the simulator.
	4. Summarize the results with a written report.


[//]: # (Image References)

[image1]: ./examples/1.nvidia_model.jpg
[image2]: ./examples/2.modified_nvidia_model.jpg
[image3]: ./examples/3.model_mean_squared_error_loss.jpeg

### Files Submitted & Code Quality
---------------------------------------------------
#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode


The Project contains the following files:

	1. model.py  : The script used to create and train the model.
	2. drive.py  : Script to drive the car autonomously.
	3. model.h5  : Trained Keras model.
	4. README.md : writeup file which descirbes the project steps.
	5. video.mp4 : A recorded video of the vehicle driving autonomously around the track.

#### 2. Submission includes functional code
Using the Udacity provided simulator and my **drive.py** file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The **model.py** file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.


### Model Architecture and Training Strategy
------------------------------------------------------------------
#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 128 (model.py lines 18-24) 

The model includes RELU layers to introduce nonlinearity (code line 20), and the data is normalized in the model using a Keras lambda layer (code line 18). 

![alt text][image1]

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 21). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 



































