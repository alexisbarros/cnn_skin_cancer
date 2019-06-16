from PIL import Image, ImageEnhance
from skimage import color, img_as_float
from sklearn.preprocessing import StandardScaler
import numpy as np
import os

# Directories array to loop for
directories = ['benign/', 'malignant/']

# Dataset and label
dataset = []

# Loop in directories
for dir in directories:

    # Loop in images
    for index, filename in enumerate(os.listdir('dataset/' + dir)):

        print(index)

        # Set label to benign (0) or malignant (1)
        if dir == 'benign/':
            label = 0
        else:
            label = 1

        # Get image
        img = Image.open('dataset/' + dir + filename)

        # Resize image to 256x256
        img = img.resize((256, 256))

        # Change brightness
        img = ImageEnhance.Brightness(img).enhance(1.5)

        # Convert img in to float and gray
        img = color.rgb2gray(img_as_float(img))

        # Normalize image and reshape
        normalizer = StandardScaler().fit(img)
        img_normalized = np.reshape(normalizer.transform(img), 65536)

        # Insert label in img array
        img_normalized = np.insert(img_normalized, 0, label)

        # Insert img into dataset array
        dataset.append(img_normalized)

# Save dataset file
np.savetxt('dataset/skin_cancer_dataset.txt', dataset)
