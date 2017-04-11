import csv
import random

import cv2
import numpy as np
from keras.models import load_model
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten, Lambda
from keras.layers.convolutional import Convolution2D, Cropping2D
from sklearn.model_selection import train_test_split
import sklearn

data_dir = "./data"
angle_correction = 0.15


"""Reads csv file from given directory"""
def read_data(dataDir, maxNum=-1):
    lines = []
    with open(dataDir + "/driving_log.csv") as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            if line[0] != "center":
                lines.append(line)
                if maxNum != -1 and len(lines) >= maxNum:
                    break
    return lines

"""Makes imagepath relative to data_dir"""
def correct_path(data_dir, path):
    return data_dir + "/IMG/" + (path.split('/')[-1])

"""Reads image in yuv format"""
def read_image(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    return img

"""The data generator"""
def generator(data_dir, samples, batch_size=32):
    num_samples = len(samples)
    while 1:
        random.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                center_img = read_image(correct_path(data_dir, batch_sample[0]))
                left_img = read_image(correct_path(data_dir, batch_sample[1]))
                right_img = read_image(correct_path(data_dir, batch_sample[2]))

                center_angle = float(batch_sample[3])

                images.extend([center_img, left_img, right_img])
                angles.extend([
                    center_angle,
                    center_angle + angle_correction,
                    center_angle - angle_correction
                ])

            augmented_images = []
            augmented_angles = []
            for (image, angle) in zip(images, angles):
                augmented_images.append(image)
                augmented_angles.append(angle)
                augmented_images.append(cv2.flip(image, 1))
                augmented_angles.append(angle * -1.0)

            X_train = np.array(augmented_images)
            y_train = np.array(augmented_angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# Reading data, different dataset for training and validation

lines = read_data(data_dir)
train_samples, validation_samples = train_test_split(lines, test_size=0.2)

# Using generators to be able to train on a big dataset

train_generator = generator(data_dir, train_samples, batch_size=48)
validation_generator = generator(data_dir, validation_samples, batch_size=48)

# Model definition like nvidia, but with wider fully connected part and dropouts

model = Sequential()

model.add(Cropping2D(cropping=((50, 20), (0, 0)), input_shape=(160, 320, 3)))
model.add(Lambda(lambda x: x / 255.0 - 0.5))
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode='valid', activation='elu'))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode='valid', activation='elu'))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode='valid', activation='elu'))
model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode='valid', activation='elu'))
model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode='valid', activation='elu'))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(512, activation='elu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='elu'))
model.add(Dropout(0.5))
model.add(Dense(32, activation='elu'))
model.add(Dense(1))

model.compile(loss="mse", optimizer="adam")

# Uncomment the next line to continue training an existing model
# model = load_model("model.h5")

# The training of the model
# The number of training samples is multiplied by 6 because the augmentation multiplies them.
model.fit_generator(
    train_generator,
    samples_per_epoch=6*len(train_samples),
    validation_data=validation_generator,
    nb_val_samples=len(validation_samples),
    nb_epoch=5)

model.save("model.h5")
