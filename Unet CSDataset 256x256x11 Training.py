import os
import keras
import tensorflow as tf
from tensorflow.python.client import device_lib
import platform
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, concatenate, Conv2DTranspose, Dropout
from CustomDataGeneratorV2 import CustomDataGenerator as cdg


SIZE_X=256
SIZE_Y=256
N_CHANNEL=1

#Quantidade de classes dos pixels
N_CLASSES=11 #classes: 0 a 10

#Informacões
print("Número de GPUs Disponiveis: ", len(tf.config.list_physical_devices('GPU')))
print("GPU: NVIDIA GeForce GTX 1070")
print("Versão Tensorflow: ",tf.__version__)
print("Versão Python: ",platform.python_version())
print("Sistema Operacional: ",platform.system(), platform.release(), "- Versão: ",platform.version())



#Instanciar gerador
idList = os.listdir(r'cityscapesdataset\csdataset\train_images\img')

training_generator = cdg(idList, batch_size=32,size_x=SIZE_X, size_y=SIZE_Y, n_channels=N_CHANNEL, n_classes=N_CLASSES, shuffle=True, 
                         path_images='cityscapesdataset\\csdataset\\train_images\img\\',
                         path_masks='cityscapesdataset\\csdataset\\train_masks\img\\')


#Definicao do modelo
def modeloUnet(n_classes, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS):

    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

    #Contraction path
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
    c1 = Dropout(0.1)(c1)
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)
    
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = Dropout(0.1)(c2)
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)
     
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)
     
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)
     
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
    
    #Expansive path 
    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
     
    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
     
    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = Dropout(0.1)(c8)
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
     
    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = Dropout(0.1)(c9)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
     
    outputs = Conv2D(n_classes, (1, 1), activation='softmax')(c9)
     
    model = Model(inputs=[inputs], outputs=[outputs])
    
    return model

#Criar Modelo
model = modeloUnet(N_CLASSES,SIZE_X,SIZE_Y,N_CHANNEL)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', tf.keras.metrics.MeanIoU(num_classes=N_CLASSES)])
model.summary()

#CallBacks
callback_ES = keras.callbacks.EarlyStopping(monitor='loss', patience=4)

checkpoint_filepath = 'checkpoints\\256x256x1-BestWeightsCSDGenerator.hdf5'
checkpoint_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath, save_weights_only=True, monitor='loss', mode='min', save_best_only=True)

#tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs")

model.fit(training_generator,epochs=50,verbose=1,callbacks=[callback_ES, checkpoint_callback])


model.save('modeloUnetCSDo256x256Generator.hdf5')
model = keras.models.load_model('modeloUnetCSDo256x256Generator.hdf5')
model.load_weights('checkpoints\\256x256x1-BestWeightsCSDGenerator.hdf5')


idListTest = os.listdir(r'cityscapesdataset\csdataset\test_images\img')

test_generator = cdg(idListTest, batch_size=32,size_x=SIZE_X, size_y=SIZE_Y, n_channels=N_CHANNEL, n_classes=N_CLASSES, shuffle=True, 
                         path_images='cityscapesdataset\\csdataset\\test_images\img\\',
                         path_masks='cityscapesdataset\\csdataset\\test_masks\img\\')


model.evaluate(test_generator, verbose=1)
