# **Behavioral Cloning** 

## Writeup

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:

* Use the simulator to collect data of good driving behavior

* Build a convolution neural network in Keras that predicts steering angles from images

* Train and validate the model with a training and validation set

* Test that the model successfully drives around track one without leaving the road

* Summarize the results with a written report


[//]: # (Image References)


[image2]: ./images/center.jpg "Center lane driving"
[image3]: ./images/far-right.jpg "Recovery Image Far right"
[image4]: ./images/right.jpg "Recovery Image right"
[image5]: ./images/center2.jpg "Recovery Image middle"
[image6]: ./images/normal.jpg "Normal Image"
[image7]: ./images/flipped.jpg "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:

* model.py containing the script to create and train the model

* drive.py for driving the car in autonomous mode

* model.h5 containing a trained convolution neural network 

* video.mp4 showing the car drive itself using my model

* README.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 and 3x3 filter sizes and depths between 24 and 64 (model.py lines 100-114) 

The model includes ELU layers to introduce nonlinearity (code line 102-106, 109, 111, 113), and the data is normalized in the model using a Keras lambda layer (code line 101). 

#### 2. Attempts to reduce overfitting in the model

The model was originally the one described in "End to End Learning for Self-Driving Cars (2016)". 
It is a very slim model (especially the fully connected part) so it did not need dropout layers or regularization. 

However to demonstrate the use of dropout layers I changed the fully connected layers to wider ones.
The model contains dropout layers in order to reduce overfitting (model.py line 108, 110, 112). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 89). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 116).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, driving in the opposite direction.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to find a model which could overfit a small dataset, and optimize that.

It was evident to use the architecture described in the article: "End to End Learning for Self-Driving Cars (2016)", because it solves basically the same problem.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had similar mean squared error on the training set as on the validation set.

But to demonstrate the use of dropout layers as mentioned in the rubric, I modified the model to a wider one, which can fit and overfit easier.

To combat the overfitting, I modified the model by adding dropout layers before and between the bigger fully connected layers.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track (eg. at the beginning of the dirt road) to improve the driving behavior in these cases, I added more data around that particular place.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 100-114) consisted of a convolution neural network with the following layers:

```
Cropping2D(cropping=((50, 20), (0, 0)), input_shape=(160, 320, 3))
Lambda(lambda x: x / 255.0 - 0.5)
Convolution2D(24, 5, 5, subsample=(2, 2), border_mode='valid', activation='elu')
Convolution2D(36, 5, 5, subsample=(2, 2), border_mode='valid', activation='elu')
Convolution2D(48, 5, 5, subsample=(2, 2), border_mode='valid', activation='elu')
Convolution2D(64, 3, 3, subsample=(1, 1), border_mode='valid', activation='elu')
Convolution2D(64, 3, 3, subsample=(1, 1), border_mode='valid', activation='elu')
Flatten()
Dropout(0.5)
Dense(512, activation='elu')
Dropout(0.5)
Dense(128, activation='elu')
Dropout(0.5)
Dense(32, activation='elu')
Dense(1)
```

To get more info about the size of each layer see [this image](images/model.png).

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. One clockwise and one counter-clockwise. Here is an example image of center lane driving:

![Center lane driving][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to go back to the middle of the road. These images show what a recovery looks like starting from the right side of the road:

![Car on the far right side][image3]
![Car on the right side][image4]
![Car in the middle of the road][image5]

To augment the data sat, I also flipped images and angles thinking that this would make the model more generic. For example, here is an image that has then been flipped:

![Original image][image6]
![Flipped image][image7]

Later I swiched most of my dataset to the one provided by udacity, as it produced a little bit better results.
I also added additional data near one problematic point.

After the collection process, I had 8000 data points. I then preprocessed this data by converting it to YUV. (Care had to be taken as the images in drive.py are RGB, but the images read by opencv are BGR.)

I also used the left and the right images with an angle correction of 3.75 degrees.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 because the driving quality of the network didn't seem to improve after that. I used an adam optimizer so that manually training the learning rate wasn't necessary.
