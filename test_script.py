from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import Sequential
from keras.preprocessing import image
import cv2
import numpy as np

# criar modelo
model = Sequential()
# Step 1: Convolution
model.add(Conv2D(32, (3, 3), input_shape=(128, 128, 3), activation='relu'))
# Step 2: Pooling
model.add(MaxPooling2D(pool_size=(2, 2)))
# Step 3: Convolution 2
model.add(Conv2D(64, (3, 3), activation='relu'))
# Step 4: Pooling 2
model.add(MaxPooling2D(pool_size=(2, 2)))
# Step 5: Flattening
model.add(Flatten())
# Step 6: Full connection
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))
# Compiling the CNN
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.load_weights('./trained_models/trained_model_cnn_2layers.h5')

# predicting images
img = image.load_img('./dataset_kaggle_lesion/validation/common_nevus/9.jpg', target_size=(128, 128))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

images = np.vstack([x])
classes = model.predict_classes(images, batch_size=32)
print(classes)
