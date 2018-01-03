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
[image3]: ./examples/3.right_2016_12_01_13_45_56_283.jpg
[image4]: ./examples/4.center_2016_12_01_13_45_56_283.jpg
[image5]: ./examples/5.left_2016_12_01_13_45_56_283.jpg

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
I have tried first to use LeNet architecture, but the vehicle was driven away from the track.
Then I have tried Nvidia architecture and the vehicle driving has been improved a lot but it still needs some modification.

**Nvidia Original Version:**

![alt text][image1]


Finally I have modified Nvidia model and used it in training my model, For details about this version, Check [Final Model Architecture](#2-final-model-architecture) 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting.
The model was trained and validated on different data sets by using 80% as training and 20% as validation to ensure that the model was not overfitting. 
The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

* The model used an adam optimizer, so the learning rate was not tuned manually ([model.py line 135](model.py#L135)).
* Number of epochs is **7**. ([model.py line 142](model.py#L142)).
* Number of batches is **32**. ([model.py line 107, line 108](model.py#L107-L108)).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I have used the training data provided by Udacity and made some preprocessing on them that help the model to train and drive autonomously in first track.

For details about how I handled the training data, see this [Creation of the Training Set & Training Process](#3-creation-of-the-training-set-&-training-process).


### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to train a model which will be able to make the vehicle drive autonomously without leaving the track

My first step was to use LeNet architecture with 3 epochs to train the model, but the vehicle went directly off the track.

My second step was to use Nvidia architecture, I thought this model might be appropriate because Nvidia solved the same problem in its end to end autonomous project.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified Nvidia architecture by adding dropout layers which helped a lot especially while driving on the bridge and in the curves as well.

Then I have preprocessed, augmented and shuffled the images in order to fit the data for my model.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track, but after adding dropout layers the vehicle was able to drive smoothly around track one.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture can be found in ([model.py line 117 to line 132](model.py#L117-L132)).

**Here's how the model looks like.**

![alt text][image2]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image20]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image30]
![alt text][image40]
![alt text][image50]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image60]
![alt text][image70]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.

