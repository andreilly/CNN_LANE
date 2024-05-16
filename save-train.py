import os
import pickle
import numpy as np
from PIL import Image

def main():
    # La cartella dove sono presenti le immagini .jpg
    directory = 'DatasetTUSimple/frames'
    # La lista per contenere le immagini convertite
    images_as_arrays = []

    # Ottenere un elenco di tutti i file .jpg nella directory
    jpg_files = [filename for filename in os.listdir(directory) if filename.lower().endswith('.jpg')]

    # Il numero totale di immagini .jpg
    total_images = len(jpg_files)

    new_size=(320,176)
    # Iterare tutte le immagini nella cartella specificata
    for index, filename in enumerate(jpg_files):
        # Caricare l'immagine
        image_path = os.path.join(directory, filename)
        with Image.open(image_path) as img:
            img_resized = img.resize(new_size)
            image_array = np.array(img_resized, np.uint8)
    
            # Aggiungere l'array convertito alla lista delle immagini
            images_as_arrays.append(image_array)

        # Calcolare e stampare la percentuale di completamento
        percentage_done = (index + 1) / total_images * 100
        print(f"Elaborazione completata: {percentage_done:.2f}% ({index + 1}/{total_images})")  # Mostra la percentuale con due cifre decimali

    # Serializzare l'array di immagini utilizzando pickle
    pickle_filename = 'train_updated.p'
    with open(pickle_filename, 'wb') as pfile:
        pickle.dump(images_as_arrays, pfile)

    print(f"Le immagini sono state salvate in {pickle_filename}")
    
main()