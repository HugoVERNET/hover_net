import os
import json
import cv2
import numpy as np

# Supprimer tous les fichiers dans le dossier 'mat'
mat_dir = 'data/resultat/mat'
if os.path.exists(mat_dir):
    for filename in os.listdir(mat_dir):
        file_path = os.path.join(mat_dir, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

# Supprimer le fichier 'qupath'
qupath_file = 'data/resultat/qupath'
if os.path.exists(qupath_file):
    os.remove(qupath_file)
    
def get_pixels_inside_contour(image, contour):
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, (255), thickness=cv2.FILLED)
    pixels = image[mask == 255]
    return pixels.tolist()

# Répertoires
json_dir = 'data/resultat/json'
image_dir = 'data/resultat/patch_non_white'
output_dir = os.path.join(json_dir, 'final')
os.makedirs(output_dir, exist_ok=True)

# Itérer sur les fichiers JSON
for filename in os.listdir(json_dir):
    if filename.endswith('.json') and not filename.endswith('-final.json'):
        json_path = os.path.join(json_dir, filename)
        with open(json_path, 'r') as file:
            data = json.load(file)
        
        # Correspondance de l'image
        base_name = os.path.splitext(filename)[0]
        image_path = os.path.join(image_dir, f'{base_name}.png')
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            print(f"Image file '{image_path}' not found.")
            continue
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Ajouter les couleurs RGB
        for key, value in data['nuc'].items():
            contour = np.array(value['contour'])
            rgb_pixels = get_pixels_inside_contour(image_rgb, contour)
            data['nuc'][key]['rgb_colors'] = rgb_pixels
        
        # Sauvegarder le JSON mis à jour
        output_path = os.path.join(output_dir, f'{base_name}-final.json')
        with open(output_path, 'w') as outfile:
            json.dump(data, outfile, indent=4)
        
        # Supprimer le fichier JSON original
        os.remove(json_path)