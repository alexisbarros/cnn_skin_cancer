from PIL import Image
from skimage import color, img_as_float, exposure
import os

# Directories array to loop for
directories = ['train/common_nevus/', 'train/melanoma/', 'validation/common_nevus/', 'validation/melanoma/']

# Loop in directories
for dir in directories:

    # Loop in images
    for index, filename in enumerate(os.listdir('dataset/' + dir)):

        print(index)

        if filename != '.DS_Store':
            # Get image
            img = Image.open('dataset/' + dir + filename)

            # Resize image to 800x800
            img = img.resize((800, 800))

            # Convert img in to float and gray
            img = color.rgb2gray(img_as_float(img))

            # Normalize image
            img_normalized = exposure.equalize_hist(img)

            # Save image normalized
            Image.fromarray((img_normalized * 255).astype('uint8'), mode='L')\
                .save('dataset_resized_normalized/' + dir + filename[:-4] + '.png')