from random import shuffle
import glob
import cv2
import numpy as np
import h5py

def ImagesToHDF5(src, h, w, hdf5_path):
    #Obtener las rutas de todas las imágenes
    paths = glob.glob(src)
    
    #Etiquetar los datos como 0=an2i, 1=tammo, 2=saavik
    labels = []
    for path in paths:
        if 'an2i' in path:
            labels.append(0)
        if 'tammo' in path:
            labels.append(1)
        if 'saavik' in path:
            labels.append(2)            

    #Barajear imágenes
    data=list(zip(paths, labels))   # utilizamos zip() para unir las rutas de las imágenes y sus etiquetas
    shuffle(data)
    
    paths, labels = zip(*data)      # *data es utilizada para separar todas las tuplas en la lista data
                                    # paths y labels se encuentran barajeados
        
    data_shape = (len(paths), h, w)
    
    #Abrir un archivo hdf5 y crear los datasets
    f=h5py.File(hdf5_path, mode='w')
    
    f.create_dataset("images", data_shape, np.uint8)    
    f.create_dataset("labels", (len(labels),), np.uint8)
    f["labels"][...]=labels
    
    for i in range(len(paths)):
        path=paths[i]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img=cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)  #Redimensionar imagen a: (h, w)
        f["images"][i, ...] = img[None]
            
    f.close()


# Cargar el dataset de datos para construir el modelo
def load_dataset(hdf5_path):
    dataset=h5py.File(hdf5_path, 'r')
    X = np.array(dataset["images"][:])
    Y = np.array(dataset["labels"][:])    
    return X, Y

