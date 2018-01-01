import cv2
import csv
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Lambda, Cropping2D, Convolution2D, MaxPooling2D, Flatten, Dense, Dropout,Activation


images_path = 'Udacity_Data/data/'
driving_data_path = 'Udacity_Data/data/driving_log.csv'


def GetDataFromExcel(DrivingDataPath):
    lines=[]
    with open (DrivingDataPath) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)
    del lines[0]        
    return lines            

def GetImagesAndSteeringData(StoredLines,ImagesPath):
    images = []
    steering = []
    correction = 0.2
    for line in StoredLines:
		  #####################Center Camera Data#######################
        center_image_name = line[0]
        center_image_path = ImagesPath + center_image_name
        center_image = cv2.imread(center_image_path)
        center_steering_value = float(line[3])
        images.append(center_image)
        steering.append(center_steering_value)
        
        #####################Left Camera Data#######################
        left_image_name = line[1].strip()
        left_image_path = ImagesPath + left_image_name
        left_image = cv2.imread(left_image_path)
        left_steering_value = center_steering_value + correction
        images.append(left_image)
        steering.append(left_steering_value)
        
        #####################right Camera Data#######################
        right_image_name = line[2].strip()
        right_image_path = ImagesPath + right_image_name
        right_image = cv2.imread(right_image_path)
        right_steering_value = center_steering_value - correction
        images.append(right_image)
        steering.append(right_steering_value)

    return images,steering

def AugmentData(TotalImages, SteeringData):
    images_augmented = []
    steering_data_augmented = []
    for image,steering in zip(TotalImages, SteeringData):
        
        images_augmented.append(image)
        images_augmented.append(cv2.flip(image,1))
        
        steering_data_augmented.append(steering)
        steering_data_augmented.append(steering*-1.0)
        
    return images_augmented, steering_data_augmented    
    

stored_lines = GetDataFromExcel(driving_data_path)

total_images, steering_data = GetImagesAndSteeringData(stored_lines,images_path)
x_train = np.array(total_images)
y_train = np.array(steering_data)

#total_images_augmented, steering_data_augmented = AugmentData(total_images, steering_data)
#x_train = np.array(total_images_augmented)
#y_train = np.array(steering_data_augmented)


# set up lambda layer
model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((50,20), (0,0))))

model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(64,3,3, activation='relu'))
model.add(Convolution2D(64,3,3, activation='relu'))
model.add(Flatten())

model.add(Dropout(p=0.5))
model.add(Activation('relu'))
model.add(Dense(100))

model.add(Dropout(p=0.5))
model.add(Activation('relu'))
model.add(Dense(50))

model.add(Dropout(p=0.5))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
history_object = model.fit(x_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=7)
model.save('model.h5')
### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()