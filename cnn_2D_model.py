from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator

# Initialize CNN
classifier = Sequential()

# Step 1: Convolution
classifier.add(Conv2D(32, (3, 3), input_shape=(128, 128, 3), activation='relu'))

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
training_set = ImageDataGenerator(
    rescale = 1. / 255,
    rotation_range = 180.,
    horizontal_flip = True
).flow_from_directory(
    'dataset_kaggle_lesion/train',
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary'
)

test_set = ImageDataGenerator(
    rescale=1. / 255,
).flow_from_directory(
    'dataset_kaggle_lesion/validation',
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary'
)

# Train the model
classifier.fit_generator(
    training_set,
    steps_per_epoch=2637 // 32,
    epochs=50,
    validation_data=test_set,
    validation_steps=660 // 32
)

# Save the model
classifier.save_weights('trained_model_rev006.h5')
