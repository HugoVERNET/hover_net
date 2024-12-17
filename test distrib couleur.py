import cv2
import json
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('c:/Users/verne/Desktop/M1/Projet annee/Test/020D.png', cv2.IMREAD_COLOR)
if image is None:
    raise FileNotFoundError("Image file '020D-hovernet.png' not found.")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Load the JSON data
with open('c:/Users/verne/Desktop/M1/Projet annee/Test/020D.json', 'r') as file:
    data = json.load(file)

# Create a mask for the contour
mask = np.zeros(image_rgb.shape[:2], dtype=np.uint8)
for key, value in data['nuc'].items():
    contour = np.array(value['contour'])
    cv2.drawContours(mask, [contour], -1, (255), thickness=cv2.FILLED)

# Get the color of each pixel inside the contour
colors = {}
for key, value in data['nuc'].items():
    contour = np.array(value['contour'])
    mask = np.zeros(image_rgb.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, (255), thickness=cv2.FILLED)
    colors[key] = image_rgb[mask == 255]

# Function to map color to gradient
def map_color_to_gradient(color):
    yellow = np.array([255, 255, 0])
    red = np.array([255, 0, 0])
    gradient = np.linspace(red, yellow, 256)
    intensity = np.mean(color)
    return gradient[int(intensity)]

# Create a new image to display the gradient colors
gradient_image = image_rgb.copy()

# Apply the gradient color to each pixel inside the contour and save the gradient colors in the JSON data
for key, color_list in colors.items():
    contour = np.array(data['nuc'][key]['contour'])
    mask = np.zeros(image_rgb.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, (255), thickness=cv2.FILLED)
    gradient_colors = [map_color_to_gradient(color).tolist() for color in image_rgb[mask == 255]]
    gradient_image[mask == 255] = np.array(gradient_colors)
    data['nuc'][key]['gradient_colors'] = gradient_colors

# Save the updated JSON data
with open('c:/Users/verne/Desktop/M1/Projet annee/Test/020D.json', 'w') as file:
    json.dump(data, file, indent=4)

# Display the original and gradient images
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(image_rgb)
plt.subplot(1, 2, 2)
plt.title('Gradient Image')
plt.imshow(gradient_image)

# Save the gradient image
output_path = 'c:/Users/verne/Desktop/M1/Projet annee/Test/gradient_image.png'
cv2.imwrite(output_path, cv2.cvtColor(gradient_image, cv2.COLOR_RGB2BGR))