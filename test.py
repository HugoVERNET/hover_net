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
patch_width = 400
patch_height = 400
white_threshold = 230  # Seuil de luminance
white_fraction = 0.8
desired_magnification = 20
overlap = 0

# Répertoires racine (à adapter)
root_input_dir = "WSI_image"  # Répertoire racine où chercher les WSI
root_output_dir = "resultat"      # Répertoire racine de sortie

# Champs du fichier CSV
csv_fields = ["patch_index", "classification", "x_start", "y_start", "x_end", "y_end"]


def find_wsi_files(root_path, extensions=(".tif", ".svs")):
    """
    Trouve tous les WSI dans l'arborescence du répertoire spécifié.
    Renvoie une liste de chemins complets vers les fichiers WSI.
    """
    wsi_files = []
    for dirpath, dirnames, filenames in os.walk(root_path):
        for filename in filenames:
            if filename.lower().endswith(extensions):
                wsi_files.append(os.path.join(dirpath, filename))
    return wsi_files

def extract_patches_with_overlap_wsi(slide, patch_w, patch_h, overlap, level, start_x=0, start_y=0):
    """
    Génère la liste des coordonnées pour extraire des patches avec chevauchement
    à partir de l'ensemble du WSI, pour un niveau donné.
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
    """
    if patch.ndim != 3 or patch.shape[2] != 3:
        if debug:
            logging.debug("Patch invalide: doit avoir 3 canaux (RGB).")
        return False, 0.0

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
    """
    try:
        objective_power = float(slide.properties.get('openslide.objective-power', 40))
    except ValueError:
        objective_power = 40

    try:
        mpp_x = float(slide.properties.get('openslide.mpp-x', 0.25))
        mpp_y = float(slide.properties.get('openslide.mpp-y', 0.25))
    except ValueError:
        mpp_x, mpp_y = 0.25, 0.25

    level_dimensions = slide.level_dimensions[level]
    level0_dimensions = slide.level_dimensions[0]

    reduction_factor_x = level0_dimensions[0] / level_dimensions[0]
    reduction_factor_y = level0_dimensions[1] / level_dimensions[1]
    reduction_factor = (reduction_factor_x + reduction_factor_y) / 2

    magnification = objective_power / reduction_factor
    return magnification

def get_level_for_magnification(slide, desired_magnification):
    """
    Retourne le niveau de résolution le plus proche de la magnification souhaitée.
    """
    levels = slide.level_count
    objective_power = float(slide.properties.get('openslide.objective-power', 40))

    magnifications = []
    for level in range(levels):
        downsample = slide.level_downsamples[level]
        magnification = objective_power / downsample
        magnifications.append(magnification)

    magnifications = np.array(magnifications)
    level = np.argmin(np.abs(magnifications - desired_magnification))
    actual_magnification = magnifications[level]
    logging.info(f"Magnification souhaitée: {desired_magnification}x, Niveau sélectionné: {level} (Magnification réelle: {actual_magnification}x)")
    return level

def is_full_patch(x_start, y_start, x_end, y_end, patch_w, patch_h):
    """
    Vérifie si un patch couvre entièrement la zone souhaitée.
    """
    return (x_end - x_start) >= patch_w and (y_end - y_start) >= patch_h

def get_mpp_from_wsi(slide):
    """
    Récupère la taille d'un pixel en microns pour un WSI.
    """
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

def process_wsi(wsi_path, output_base_dir):
    """
    Traite une WSI unique et enregistre les patches dans la structure A et B.
    """
    try:
        if not os.path.isfile(wsi_path):
            logging.error(f"Le fichier {wsi_path} n'existe pas.")
            return
        slide = openslide.OpenSlide(wsi_path)
    except openslide.OpenSlideUnsupportedFormatError:
        logging.error(f"Le fichier {wsi_path} n'est pas supporté par OpenSlide.")
        return
    except Exception as e:
        logging.error(f"Erreur lors de l'ouverture du fichier WSI : {e}")
        return

    try:
        mpp = get_mpp_from_wsi(slide)
        if mpp:
            logging.info(f"La taille d'un pixel est d'environ {mpp:.4f} µm.")
            # Ajustement du chevauchement (exemple, à adapter si besoin)
            # Ici, c'est un exemple de calcul basé sur votre code source.
            overlap = int(round((15*2)/ (mpp *2)))
            logging.info(f"Chevauchement ajusté : {overlap}")
        else:
            overlap = 0
    except Exception as e:
        logging.error(f"Erreur lors de la récupération du MPP : {e}")
        overlap = 0

    # Déterminer le niveau pour la magnification souhaitée
    level = get_level_for_magnification(slide, desired_magnification)
    coords, (W, H) = extract_patches_with_overlap_wsi(slide, patch_width, patch_height, overlap, level, start_x=0, start_y=0)
    logging.info(f"Dimensions du niveau {level} : {W}x{H}")
    logging.info(f"Nombre de patches à extraire : {len(coords)}")

    magnification_selected_level = get_magnification(slide, level=level)
    logging.info(f"Magnification au niveau {level} : {magnification_selected_level}x")

    # Construction des répertoires de sortie
    # On reproduit la structure d'entrée dans la sortie
    relative_path = os.path.relpath(wsi_path, start=root_input_dir)
    wsi_output_dir = os.path.join(output_base_dir, os.path.splitext(relative_path)[0])  # Sans extension
    os.makedirs(wsi_output_dir, exist_ok=True)

    output_dir_non_white = os.path.join(wsi_output_dir, "A")
    output_dir_white = os.path.join(wsi_output_dir, "B")
    os.makedirs(output_dir_non_white, exist_ok=True)
    os.makedirs(output_dir_white, exist_ok=True)

    # Fichier CSV spécifique à cette WSI
    csv_filename = os.path.join(wsi_output_dir, "patch_classification.csv")

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

            if not is_full_patch(x_start, y_start, x_end, y_end, patch_width, patch_height):
                logging.debug(f"Patch {i} est incomplet. Ignoré.")
                continue

            try:
                region_pil = slide.read_region((x_start, y_start), level, (width, height))
            except Exception as e:
                logging.error(f"Erreur lors de l'extraction du patch {i} : {e}")
                continue

            region_pil = region_pil.convert("RGB")
            patch_np = np.array(region_pil)
            debug = i < 5
            is_white, fraction_white = is_patch_white_luminance(patch_np, threshold=white_threshold, white_fraction=white_fraction, debug=debug)

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
                patch_filename_white = f"patch_white_{i}.png"
                try:
                    region_pil.save(os.path.join(output_dir_white, patch_filename_white))
                    saved_white_patches += 1
                    if debug:
                        first_patches_white.append(patch_np)
                except Exception as e:
                    logging.error(f"Erreur lors de la sauvegarde du patch blanc {i} : {e}")
                continue

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


    logging.info(f"Traitement terminé pour {wsi_path}. Les patches non-blancs sont dans '{output_dir_non_white}', les patches blancs dans '{output_dir_white}'.")
    logging.info(f"Les informations de classification sont enregistrées dans '{csv_filename}'.")

def main():
    wsi_files = find_wsi_files(root_input_dir)

    if not wsi_files:
        logging.info("Aucun fichier WSI trouvé.")
        return

    for wsi_path in wsi_files:
        logging.info(f"Traitement du WSI : {wsi_path}")
        process_wsi(wsi_path, root_output_dir)

if __name__ == "__main__":
    main()
