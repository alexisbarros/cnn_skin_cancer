from PIL import Image
from skimage import color, img_as_float
import numpy as np
import os

path_from = './PATH_FROM/'
path_to = './PATH_TO/'

# array de diretorios
directories = ['train/common_nevus/', 'validation/common_nevus/', 'train/melanoma/', 'validation/melanoma/']

# Dataset e label
dataset = []
label = 0
count = 0

# Loop em diretorios
for dir in directories:

    # Loop nas imagens
    for index, filename in enumerate(os.listdir(path_from + dir)):

        print(index)

        # Get image
        img = Image.open(path_from + dir + filename)

        # Convert img in to float and gray
        img = img_as_float(img)

        # Insert label in img array
        img = np.insert(img, 0, label)

        # Insert img into dataset array
        dataset.append(img)

    if count == 1:
        label = 1

    count += 1

# Save dataset file
np.savetxt(path_to + 'skin_cancer_dataset.txt', dataset)
