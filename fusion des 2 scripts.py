import openslide
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import logging


# Charger les données JSON
with open('c:/Users/verne/Desktop/M1/Projet annee/Test/020D.json', 'r') as file:
    data = json.load(file)

slide = openslide.OpenSlide("C:/Users/verne/Desktop/M1/Projet annee/WSI-segmentation/HES/020D.tif")

def get_mpp_from_wsi(slide):
    try:
        mpp_x = float(slide.properties.get('openslide.mpp-x', 0))
        mpp_y = float(slide.properties.get('openslide.mpp-y', 0))
        if mpp_x > 0 and mpp_y > 0:
            if abs(mpp_x - mpp_y) > 1e-3:
                logging.warning(f"Différence notable entre mpp-x ({mpp_x}) et mpp-y ({mpp_y}).")
            return (mpp_x + mpp_y) / 2
        elif mpp_x > 0:
            return mpp_x
        elif mpp_y > 0:
            return mpp_y
        else:
            logging.error("Propriété mpp-x ou mpp-y introuvable dans le fichier WSI.")
            return None
    except ValueError as e:
        logging.error(f"Erreur lors de la récupération des mpp : {e}")
        return None

mpp = get_mpp_from_wsi(slide)
print(f"La taille d'un pixel est d'environ {mpp:.4f} µm.")

def shoelace_formula(x, y):
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

areas = []
perimeters = []
roundness_values = []

for key, value in data['nuc'].items():
    contour = value['contour']
    x_coords = [point[0] for point in contour]
    y_coords = [point[1] for point in contour]
    
    area_pixels = shoelace_formula(np.array(x_coords), np.array(y_coords))
    area_microns = area_pixels * (mpp ** 2)
    areas.append(area_microns)
    
    perimeter_pixels = np.sum(np.sqrt(np.diff(x_coords)**2 + np.diff(y_coords)**2))
    perimeter_microns = perimeter_pixels * mpp
    perimeters.append(perimeter_microns)
    
    roundness = (4 * np.pi * area_microns) / (perimeter_microns ** 2)
    roundness_values.append(roundness)
    
    data['nuc'][key]['area_microns'] = area_microns
    data['nuc'][key]['roundness'] = roundness
    
    print(f'Area of Nucleus {key}: {area_microns:.2f} µm²')
    print(f'Roundness of Nucleus {key}: {roundness:.2f}')

mean_area = np.mean(areas)
variance_area = np.var(areas)
std_dev_area = np.std(areas)

print(f'Mean Area: {mean_area:.2f} µm²')
print(f'Variance of Area: {variance_area:.2f} µm²')
print(f'Standard Deviation of Area: {std_dev_area:.2f} µm²')

mean_roundness = np.mean(roundness_values)
variance_roundness = np.var(roundness_values)
std_dev_roundness = np.std(roundness_values)

print(f'Mean Roundness: {mean_roundness:.2f}')
print(f'Variance of Roundness: {variance_roundness:.2f}')
print(f'Standard Deviation of Roundness: {std_dev_roundness:.2f}')

summary = {
    'mean_area': mean_area,
    'variance_area': variance_area,
    'std_dev_area': std_dev_area,
    'mean_roundness': mean_roundness,
    'variance_roundness': variance_roundness,
    'std_dev_roundness': std_dev_roundness
}

data['summary'] = summary

# Write results to JSON file
output_path = 'c:/Users/verne/Desktop/M1/Projet annee/Test/020D.json'
with open(output_path, 'w') as outfile:
    json.dump(data, outfile, indent=4)


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