from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import Sequential
import numpy as np
from keras.preprocessing import image
import os

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

model.load_weights('./trained_model_cnn_2layers.h5')

# variaveis falso e verdadeiro, positivo e negativo
tp = 0
fn = 0
tn = 0
fp = 0

# array de diretorios
directories = ['common_nevus/', 'melanoma/']
flag = 0

# Loop em diretorios
for dir in directories:

    # Loop nas imagens
    for index, filename in enumerate(os.listdir('dataset/validation/' + dir)):
        # classificando as imagens
        img = image.load_img('./dataset/validation/' + dir + filename, target_size=(128, 128))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        images = np.vstack([x])
        classes = model.predict_classes(images, batch_size=32)

        print('real_class:', flag, '| pred_class:', classes[0][0])

        if flag == 0:
            if (classes[0][0] == 0):
                tn += 1
            else:
                fp += 1

        else:
            if (classes[0][0] == 1):
                tp += 1
            else:
                fn += 1

    flag = 1

# imprimir resultados
print('tp', tp)
print('tn', tn)
print('fp', fp)
print('fn', fn)
print('----')
print('sensitivity: ', "%.2f" % (tp/(tp+fn)))
print('specificity: ', "%.2f" % (tn/(tn+fp)))
