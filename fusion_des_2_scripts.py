import os
import json
import cv2
import numpy as np

def get_pixels_inside_contour(image, contour):
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, (255), thickness=cv2.FILLED)
    pixels = image[mask == 255]
    return pixels.tolist()

# Load JSON data
json_path = 'C:/Users/verne/Documents/GitHub/hover_net2/resultat/json/020D.json'
with open(json_path, 'r') as file:
    data = json.load(file)

# Load image
image_path = 'C:/Program Files/Git/image/020D.png'
image = cv2.imread(image_path, cv2.IMREAD_COLOR)
if image is None:
    raise FileNotFoundError(f"Image file '{image_path}' not found.")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Process each contour and add RGB information to JSON
for key, value in data['nuc'].items():
    contour = np.array(value['contour'])
    rgb_pixels = get_pixels_inside_contour(image_rgb, contour)
    data['nuc'][key]['rgb_colors'] = rgb_pixels

# Ensure the directory exists
output_dir = 'C:/Users/verne/Documents/GitHub/hover_net2/resultat/json'
os.makedirs(output_dir, exist_ok=True)

# Write updated results to JSON file
output_path = os.path.join(output_dir, 'output.json')
with open(output_path, 'w') as outfile:
    json.dump(data, outfile, indent=4)