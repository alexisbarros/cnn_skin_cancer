from PIL import Image, ImageEnhance
from skimage import color, img_as_float
from sklearn.preprocessing import StandardScaler, Normalizer
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

        # Get image
        img = Image.open('dataset/' + dir + filename)

        # Resize image to 256x256
        img = img.resize((256, 256))

        # Change brightness
        img = ImageEnhance.Brightness(img).enhance(1.8)

        # Convert img in to float and gray
        img = color.rgb2gray(img_as_float(img))

        img_to_save = Image.fromarray(img / (img.max() / 255.0)).convert('RGB')
        img_to_save.save('dataset_normalized/' + dir + filename + '.png')
