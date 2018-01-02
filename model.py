import cv2
import csv
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Lambda, Cropping2D, Convolution2D, MaxPooling2D, Flatten, Dense, Dropout, Activation


images_path = 'Udacity_Data/data/'
driving_data_path = 'Udacity_Data/data/driving_log.csv'

def GetDataFromExcel(DrivingDataPath):
    lines=[]
    print("Reading Data from Excel....")
    with open (DrivingDataPath) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)
    del lines[0]
    print("Done")
    print()    
    return lines            

def GetImagesAndSteeringData(StoredLines,ImagesPath):
    images = []
    steering = []
    correction = 0.2
    print("Getting Camera Data....")
    for line in StoredLines:
        
		#####################Center Camera Data#######################
		# Get center Image.
		center_image_name = line[0]
        center_image_path = ImagesPath + center_image_name
        center_image = cv2.imread(center_image_path)
		# Get center image steer angle.
        center_steering_value = float(line[3])
		# Add center image to images array.
        images.append(center_image)
        # Add steer angle of center image to steering array.
		steering.append(center_steering_value)
        
        #####################Left Camera Data#######################
        # Get left Image.
		left_image_name = line[1].strip()
        left_image_path = ImagesPath + left_image_name
        left_image = cv2.imread(left_image_path)
		# Get left image steer angle and modify it with the correction value.
        left_steering_value = center_steering_value + correction
        # Add left image to images array.
		images.append(left_image)
		# Add steer angle of left image to steering array.
        steering.append(left_steering_value)
        
        #####################Right Camera Data#######################
        # Get right Image.
		right_image_name = line[2].strip()
        right_image_path = ImagesPath + right_image_name
        right_image = cv2.imread(right_image_path)
		# Get right image steer angle and modify it with the correction value.
        right_steering_value = center_steering_value - correction
		# Add right image to images array.
        images.append(right_image)
		# Add steer angle of right image to steering array.
        steering.append(right_steering_value)
        
    print ("Camera Data Captured")
    print()
    return images,steering   
    
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        samples = sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for imagePath, measurement in batch_samples:
                images.append(imagePath)
                angles.append(measurement)
                # Flipping the data
                images.append(cv2.flip(imagePath,1))
                angles.append(measurement*-1.0)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)
            
# Read Data from driving_log.            
stored_lines = GetDataFromExcel(driving_data_path)

# Get Images and Steering Data for each image type (Center,Left,Right)
total_images, steering_data = GetImagesAndSteeringData(stored_lines,images_path)

# Splitting samples and creating generators.
samples = list(zip(total_images, steering_data))
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

print('Train samples: {}'.format(len(train_samples)))
print('Validation samples: {}'.format(len(validation_samples)))
print()

train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)


# set up lambda layer
model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((50,20), (0,0))))

# Modified version of Nvidia Model
model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(64,3,3, activation='relu'))
model.add(Convolution2D(64,3,3, activation='relu'))
model.add(Flatten())

model.add(Dropout(p=0.3))
model.add(Dense(100, activation='relu'))

model.add(Dropout(p=0.3))
model.add(Dense(50, activation='relu'))

model.add(Dropout(p=0.3))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

# Compile the model 
model.compile(loss='mse', optimizer='adam')
print("Training the model")
print()

# Train the model 
history_object = model.fit_generator(train_generator, samples_per_epoch= \
                 len(train_samples), validation_data=validation_generator, \
                 nb_val_samples=len(validation_samples), nb_epoch=7, verbose=1)

# Save model				 
model.save('model.h5')
print("Model Saved")
print()

# plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()