from PIL import Image
from skimage import color, img_as_float
from sklearn.preprocessing import Normalizer
import numpy as np
import os

# Directories array to loop for
directories = ['common_nevus/', 'melanoma/']

# Dataset and label
dataset = []
label = 0

# Loop in directories
for dir in directories:

    # Loop in images
    for index, filename in enumerate(os.listdir('dataset/' + dir)):

        print(index)

        if filename != '.DS_Store':
            # Get image
            img = Image.open('dataset/' + dir + filename)

            # Resize image to 128x128
            img = img.resize((128, 128))

            # Convert img in to float and gray
            img = color.rgb2gray(img_as_float(img))

            # Normalize image and reshape
            normalizer = Normalizer().fit(img)
            img_normalized = np.reshape(normalizer.transform(img), 16384)

            # Insert label in img array
            img_normalized = np.insert(img_normalized, 0, label)

            # Insert img into dataset array
            dataset.append(img_normalized)

    label = 1

# Save dataset file
np.savetxt('dataset/skin_cancer_dataset.txt', dataset)
