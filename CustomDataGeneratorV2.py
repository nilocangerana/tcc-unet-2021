import numpy as np
from keras.utils import to_categorical ,Sequence, normalize
import cv2 

class CustomDataGenerator(Sequence):
    def __init__(self, list_IDs, batch_size=36, size_x=256, size_y=256, n_channels=1, n_classes=11, shuffle=True, path_images='', path_masks=''):
        self.size_x=size_x
        self.size_y=size_y
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.path_images=path_images
        self.path_masks=path_masks
        self.on_epoch_end()

    def __len__(self):
        #numero de batchs por epoch
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        X, y = self.__data_generation(list_IDs_temp)
        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        #Gerar dados contendo batch_size amostras
        # inicializacao dos arrays
        X = np.empty((self.batch_size,self.size_x, self.size_y))
        y = np.empty((self.batch_size,self.size_x, self.size_y))

        # Gerar os dados
        for i, ID in enumerate(list_IDs_temp):
            # armazena imagens
            X[i,] = cv2.imread(self.path_images + ID,0)

            # armazena mascaras
            y[i,] = cv2.imread(self.path_masks + ID,0)

        return normalize(np.expand_dims(X, axis=3),axis=1), to_categorical(y, num_classes=self.n_classes)