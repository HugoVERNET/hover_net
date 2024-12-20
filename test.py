import openslide
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import logging
import csv
 
# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Paramètres ajustés
wsi_path = "C:/Users/verne/Desktop/M1/Projet annee/WSI-segmentation/HES/020D.tif"  # Remplacez par le chemin de votre WSI
patch_width = 400
patch_height = 400

white_threshold = 230  # Seuil de luminance ajusté
white_fraction = 0.8    # Fraction minimale ajustée
desired_magnification = 20  # Magnification souhaitée
overlap = 0


# Dossiers de sortie
output_dir_non_white = "C:/Users/verne/Documents/GitHub/hover_net2/resultat/patch_non_white"  # Dossier pour les patches non-blancs
output_dir_white = "C:/Users/verne/Documents/GitHub/hover_net2/resultat/patch_white"      # Dossier pour les patches blancs
os.makedirs(output_dir_non_white, exist_ok=True)
os.makedirs(output_dir_white, exist_ok=True)

# Fichier de log CSV
csv_filename = "patch_classification.csv"
csv_fields = ["patch_index", "classification", "x_start", "y_start", "x_end", "y_end"]

def extract_patches_with_overlap_wsi(slide, patch_w, patch_h, overlap, level, start_x=0, start_y=0):
    """
    Génère la liste des coordonnées pour extraire des patches avec chevauchement
    à partir de l'ensemble du WSI, en utilisant un niveau de résolution spécifique.
    
    Args:
        slide (openslide.OpenSlide): L'objet WSI.
        patch_w (int): Largeur du patch.
        patch_h (int): Hauteur du patch.
        overlap (int): Chevauchement entre les patches.
        level (int): Niveau de résolution à utiliser.
        start_x (int, optionnel): Position de départ sur l'axe X. Défaut à 0.
        start_y (int, optionnel): Position de départ sur l'axe Y. Défaut à 0.

    Returns:
        list: Liste des coordonnées pour l'extraction des patches.
        tuple: (largeur, hauteur) du niveau spécifié.
    """
    w, h = slide.level_dimensions[level]
    coords = []
    for y in range(start_y, h, patch_h):
        for x in range(start_x, w, patch_w):
            x_start = max(x - overlap, 0)
            y_start = max(y - overlap, 0)
            x_end = min(x + patch_w + overlap, w)
            y_end = min(y + patch_h + overlap, h)
            coords.append((x_start, y_start, x_end, y_end))
    return coords, (w, h)

def is_patch_white_luminance(patch, threshold=230, white_fraction=0.8, debug=False):
    """
    Vérifie si un patch est majoritairement blanc basé sur la luminance.

    Args:
        patch (np.ndarray): Le patch sous forme de tableau NumPy (H, W, 3).
        threshold (int, optionnel): Seuil pour la luminance des pixels blancs. Défaut à 230.
        white_fraction (float, optionnel): Fraction minimale de pixels blancs requise. Défaut à 0.8 (80%).
        debug (bool, optionnel): Si True, affiche des informations de débogage. Défaut à False.

    Returns:
        bool: True si le patch est majoritairement blanc, False sinon.
        float: Fraction de pixels blancs dans le patch.
    """
    if patch.ndim != 3 or patch.shape[2] != 3:
        if debug:
            logging.debug("Patch invalide: doit avoir 3 canaux (RGB).")
        return False, 0.0

    # Calculer la luminance en utilisant la formule standard
    luminance = 0.2126 * patch[:, :, 0] + 0.7152 * patch[:, :, 1] + 0.0722 * patch[:, :, 2]

    if debug:
        logging.debug(f"Luminance min: {luminance.min()}, Luminance max: {luminance.max()}")
        logging.debug(f"Fraction de pixels avec luminance >= {threshold} : {(luminance >= threshold).mean():.2f}")

    white_pixels = luminance >= threshold
    fraction_white = np.sum(white_pixels) / luminance.size

    if debug:
        logging.debug(f"Fraction de pixels blancs dans le patch : {fraction_white:.2f}")

    return fraction_white >= white_fraction, fraction_white

def get_magnification(slide, level=0):
    """
    Obtient la magnification effective d'une WSI pour un niveau donné.

    Args:
        slide (openslide.OpenSlide): L'objet WSI.
        level (int, optionnel): Niveau de résolution. Défaut à 0.

    Returns:
        float: Magnification effective.
    """
    try:
        objective_power = float(slide.properties.get('openslide.objective-power', 40))  # Valeur par défaut 40x
    except ValueError:
        objective_power = 40  # Valeur par défaut si la propriété n'est pas disponible

    # Obtenir la résolution en microns par pixel au niveau 0
    try:
        mpp_x = float(slide.properties.get('openslide.mpp-x', 0.25))  # Valeur par défaut 0.25 µm/pixel
        mpp_y = float(slide.properties.get('openslide.mpp-y', 0.25))
    except ValueError:
        mpp_x, mpp_y = 0.25, 0.25  # Valeur par défaut

    # Obtenir les dimensions du niveau 0 et du niveau spécifié
    level_dimensions = slide.level_dimensions[level]
    level0_dimensions = slide.level_dimensions[0]
    
    # Calculer le facteur de réduction entre les niveaux
    reduction_factor_x = level0_dimensions[0] / level_dimensions[0]
    reduction_factor_y = level0_dimensions[1] / level_dimensions[1]
    reduction_factor = (reduction_factor_x + reduction_factor_y) / 2  # Moyenne des facteurs

    # Calculer la magnification effective
    magnification = objective_power / reduction_factor

    return magnification

def get_level_for_magnification(slide, desired_magnification):
    """
    Retourne le niveau de résolution qui correspond le mieux à la magnification désirée.
    
    Args:
        slide (openslide.OpenSlide): L'objet WSI.
        desired_magnification (float): Magnification souhaitée (ex. 20).
    
    Returns:
        int: Niveau de résolution le plus proche de la magnification désirée.
    """
    levels = slide.level_count
    objective_power = float(slide.properties.get('openslide.objective-power', 40))  # Valeur par défaut 40x

    # Calculer le facteur de magnification pour chaque niveau
    magnifications = []
    for level in range(levels):
        downsample = slide.level_downsamples[level]
        magnification = objective_power / downsample
        magnifications.append(magnification)

    # Trouver le niveau avec la magnification la plus proche
    magnifications = np.array(magnifications)
    level = np.argmin(np.abs(magnifications - desired_magnification))
    actual_magnification = magnifications[level]
    logging.info(f"Magnification souhaitée: {desired_magnification}x, Niveau sélectionné: {level} (Magnification réelle: {actual_magnification}x)")
    return level

def is_full_patch(x_start, y_start, x_end, y_end, patch_w, patch_h):
    """
    Vérifie si un patch couvre entièrement la zone souhaitée.

    Args:
        x_start (int): Coordonnée de départ sur l'axe X.
        y_start (int): Coordonnée de départ sur l'axe Y.
        x_end (int): Coordonnée de fin sur l'axe X.
        y_end (int): Coordonnée de fin sur l'axe Y.
        patch_w (int): Largeur du patch.
        patch_h (int): Hauteur du patch.

    Returns:
        bool: True si le patch est complet, False sinon.
    """
    return (x_end - x_start) >= patch_w and (y_end - y_start) >= patch_h



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


def main():
    # Ouvre le WSI avec gestion des exceptions
    try:
        if not os.path.isfile(wsi_path):
            logging.error(f"Le fichier {wsi_path} n'existe pas.")
            return
        slide = openslide.OpenSlide(wsi_path)
    except openslide.OpenSlideUnsupportedFormatError:
        logging.error(f"Le fichier {wsi_path} n'est pas supporté par OpenSlide.")
        return
    except FileNotFoundError:
        logging.error(f"Le fichier {wsi_path} n'a pas été trouvé.")
        return
    except Exception as e:
        logging.error(f"Erreur lors de l'ouverture du fichier WSI : {e}")
        return
    
    try:
        mpp = get_mpp_from_wsi(slide)
        if mpp:
            print(f"La taille d'un pixel est d'environ {mpp:.4f} µm.")
            overlap = int(round((15*2)/ (mpp *2)))
            print(f"Chevauchement ajusté : {overlap}")
        
    except Exception as e:
        print(f"Erreur lors de l'ouverture du WSI : {e}")

    # Déterminer le niveau pour la magnification souhaitée
    level = get_level_for_magnification(slide, desired_magnification)
    coords, (W, H) = extract_patches_with_overlap_wsi(slide, patch_width, patch_height, overlap, level, start_x=0, start_y=0)
    logging.info(f"Dimensions du niveau {level} : {W}x{H}")
    logging.info(f"Nombre de patches à extraire : {len(coords)}")
    
    # Obtenir la magnification au niveau sélectionné
    magnification_selected_level = get_magnification(slide, level=level)
    logging.info(f"Magnification au niveau {level} : {magnification_selected_level}x")
    
    # Initialiser le fichier CSV
    with open(csv_filename, mode='w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
        writer.writeheader()
    
        first_patches_non_white = []
        first_patches_white = []
        saved_non_white_patches = 0
        saved_white_patches = 0

        for i, (x_start, y_start, x_end, y_end) in enumerate(coords):
            width = x_end - x_start
            height = y_end - y_start

            # Vérification des patches complets
            if not is_full_patch(x_start, y_start, x_end, y_end, patch_width, patch_height):
                logging.debug(f"Patch {i} est incomplet. Ignoré.")
                continue

            # Extrait le patch du WSI
            try:
                region_pil = slide.read_region((x_start, y_start), level, (width, height))
            except Exception as e:
                logging.error(f"Erreur lors de l'extraction du patch {i} : {e}")
                continue

            # Convertir le patch en RGB
            region_pil = region_pil.convert("RGB")
            patch_np = np.array(region_pil)

            # Vérifie si le patch est majoritairement blanc avec débogage pour les premiers patches
            debug = i < 5  # Activer le débogage pour les 5 premiers patches
            is_white, fraction_white = is_patch_white_luminance(patch_np, threshold=white_threshold, white_fraction=white_fraction, debug=debug)

            # Enregistrement dans le fichier CSV
            classification = "blanc" if is_white else "non_blanc"
            writer.writerow({
                "patch_index": i,
                "classification": classification,
                "x_start": x_start,
                "y_start": y_start,
                "x_end": x_end,
                "y_end": y_end
            })

            if is_white:
                # Sauvegarde le patch blanc dans "B"
                patch_filename_white = f"patch_white_{i}.png"
                try:
                    region_pil.save(os.path.join(output_dir_white, patch_filename_white))
                    saved_white_patches += 1
                    if debug:
                        first_patches_white.append(patch_np)
                        plt.figure(figsize=(4,4))
                        plt.imshow(patch_np)
                        plt.title(f"Patch Blanc {i}")
                        plt.axis('off')
                        plt.show()
                except Exception as e:
                    logging.error(f"Erreur lors de la sauvegarde du patch blanc {i} : {e}")
                continue  # Passer à l'itération suivante

            # Sauvegarde le patch non blanc dans "A"
            patch_filename_non_white = f"patch_non_white_{i}.png"
            try:
                region_pil.save(os.path.join(output_dir_non_white, patch_filename_non_white))
                saved_non_white_patches += 1
                if debug and len(first_patches_non_white) < 7:
                    first_patches_non_white.append(patch_np)
            except Exception as e:
                logging.error(f"Erreur lors de la sauvegarde du patch non blanc {i} : {e}")
                continue

    slide.close()

    logging.info(f"Nombre de patches non-blancs sauvegardés dans '{output_dir_non_white}': {saved_non_white_patches}")
    logging.info(f"Nombre de patches blancs sauvegardés dans '{output_dir_white}': {saved_white_patches}")
    logging.info(f"Magnification effective des patches sauvegardés : {magnification_selected_level}x")


    
    

    logging.info("Traitement terminé. Les patches non-blancs sont enregistrés dans le dossier 'A' et les patches blancs dans le dossier 'B' au format PNG.")
    logging.info(f"Les informations de classification sont enregistrées dans '{csv_filename}'.")
    
if __name__ == "__main__":
    main()

