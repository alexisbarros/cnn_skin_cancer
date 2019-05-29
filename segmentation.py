from PIL import Image
import os

directories = ['train/common_nevus', 'train/melanoma',
               'validation/common_nevus', 'validation/melanoma']

for dir in directories:
    for filename in os.listdir(dir):
        if filename != '.DS_Store':
            # Get image and convert to grayscale
            img = Image.open(dir + '/' + filename).convert('LA')
            img.save('data_grayscale/' + dir + '/' + filename[:-3] + 'png')
