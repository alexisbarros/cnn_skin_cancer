# Import packages
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator

# Initialize CNN
classifier = Sequential()

# Step 1: Convolution
classifier.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))

# Step 2: Pooling
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Step 3: Convolution 2
classifier.add(Conv2D(64, (3, 3), activation='relu'))

# Step 4: Pooling 2
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Step 5: Flattening
classifier.add(Flatten())

# Step 6: Full connection
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dense(units=1, activation='sigmoid'))

# Compiling the CNN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fitting the CNN to the images
train_datagen = ImageDataGenerator(
  rescale=1./255,
  shear_range=0.2,
  zoom_range=0.2,
  horizontal_flip=True
)
test_datagen = ImageDataGenerator(
  rescale=1./255
)
training_set = train_datagen.flow_from_directory(
  'data_grayscale/train',
  target_size=(64, 64),
  batch_size=32,
  class_mode='binary'
)
test_set = test_datagen.flow_from_directory(
  'data_grayscale/validation',
  target_size=(64, 64),
  batch_size=32,
  class_mode='binary'
)

# Train the model
classifier.fit_generator(
  training_set,
  steps_per_epoch=480,
  epochs=5,
  validation_data=test_set,
  validation_steps=48
)

# Save the model
classifier.save_weights('trained_model.h5')
