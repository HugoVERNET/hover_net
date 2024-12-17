import openslide
import json
import numpy as np
import matplotlib.pyplot as plt

# Charger les données JSON
with open('c:/Users/verne/Desktop/M1/Projet annee/Test graph/020D.json', 'r') as file:
    data = json.load(file)

slide = openslide.OpenSlide("C:/Users/verne/Desktop/M1/Projet annee/WSI-segmentation/HES/020D.tif")

def get_mpp_from_wsi(slide):
    """
    Récupère la taille d'un pixel en microns pour un WSI à sa magnification de base.
   
    Args:
        slide (openslide.OpenSlide): L'objet WSI chargé.
   
    Returns:
        float: La taille d'un pixel en microns (mpp-x ou mpp-y).
    """
    try:
        mpp_x = float(slide.properties.get('openslide.mpp-x', 0))  # Taille en microns par pixel, axe X
        mpp_y = float(slide.properties.get('openslide.mpp-y', 0))  # Taille en microns par pixel, axe Y
       
        # Vérification si les deux valeurs sont cohérentes
        if mpp_x > 0 and mpp_y > 0:
            if abs(mpp_x - mpp_y) > 1e-3:  # Vérifie s'il y a une grande différence
                logging.warning(f"Différence notable entre mpp-x ({mpp_x}) et mpp-y ({mpp_y}).")
            return (mpp_x + mpp_y) / 2  # Moyenne des deux valeurs
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

# Calculer et afficher l'aire et le périmètre de chaque contour en microns carrés et microns
areas = []
perimeters = []
roundness_values = []
for key, value in data['nuc'].items():
    contour = value['contour']
    x_coords = [point[0] for point in contour]
    y_coords = [point[1] for point in contour]
    
    area_pixels = shoelace_formula(np.array(x_coords), np.array(y_coords))
    area_microns = area_pixels * (mpp ** 2)  # Conversion de pixels carrés à microns carrés
    areas.append(area_microns)
    
    # Calcul du périmètre
    perimeter_pixels = np.sum(np.sqrt(np.diff(x_coords)**2 + np.diff(y_coords)**2))
    perimeter_microns = perimeter_pixels * mpp  # Conversion de pixels à microns
    perimeters.append(perimeter_microns)
    
    # Calcul de la roundness
    roundness =  (4 * np.pi * area_microns)/(perimeter_microns ** 2)
    roundness_values.append(roundness)
    
    print(f'Area of Nucleus {key}: {area_microns:.2f} µm²')
    print(f'Roundness of Nucleus {key}: {roundness:.2f}')

# Calculer la moyenne et la variance des aires
mean_area = np.mean(areas)
variance_area = np.var(areas) 
std_dev_area = np.std(areas)

print(f'Mean Area: {mean_area:.2f} µm²')
print(f'Variance of Area: {variance_area:.2f} µm²')
print(f'Standard Deviation of Area: {std_dev_area:.2f} µm²')

# Calculer la moyenne et la variance des roundness
mean_roundness = np.mean(roundness_values)
variance_roundness = np.var(roundness_values)
std_dev_roundness = np.std(roundness_values)

print(f'Mean Roundness: {mean_roundness:.2f}')
print(f'Variance of Roundness: {variance_roundness:.2f}')
print(f'Standard Deviation of Roundness: {std_dev_roundness:.2f}')