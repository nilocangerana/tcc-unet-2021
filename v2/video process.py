import os
import cv2 
import numpy as np
from tqdm import tqdm
import keras
import tensorflow as tf
from tensorflow.python.client import device_lib
import platform
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, concatenate, Conv2DTranspose, Dropout, Add
from keras.utils import normalize
import math



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


def modeloVgg16(n_classes, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS):

    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

    c1 = Conv2D(16, (3,3),activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
    c1 = Conv2D(16, (3,3),activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)
    
    c2 = Conv2D(32, (3,3),activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = Conv2D(32, (3,3),activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)
    
    c3 = Conv2D(64, (3,3),activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = Conv2D(64, (3,3),activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    c3 = Conv2D(64, (3,3),activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)
    
    c4 = Conv2D(128, (3,3),activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = Conv2D(128, (3,3),activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    c4 = Conv2D(128, (3,3),activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = MaxPooling2D((2, 2))(c4)
    
    c5 = Conv2D(128, (3,3),activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = Conv2D(128, (3,3),activation='relu', kernel_initializer='he_normal', padding='same')(c5)
    c5 = Conv2D(128, (3,3),activation='relu', kernel_initializer='he_normal', padding='same')(c5)
    p5 = MaxPooling2D((2, 2))(c5)
    
    c6 = Conv2D(1024, (7,7),activation='relu', kernel_initializer='he_normal', padding='same')(p5)
    c6 = Dropout(0.5)(c6)
    c7 = Conv2D(1024, (1,1),activation='relu', kernel_initializer='he_normal', padding='same')(c6)
    c7 = Dropout(0.5)(c7)
    
    p4_n = Conv2D(n_classes, (1, 1), activation='relu',kernel_initializer='he_normal', padding='same')(p4)
    u2 = Conv2DTranspose(n_classes, kernel_size=(2, 2), strides=(2, 2), padding='same')(c7)
    u2_add = Add()([p4_n, u2])
    
    p3_n = Conv2D(n_classes, (1, 1), activation='relu',kernel_initializer='he_normal', padding='same')(p3)
    u4 = Conv2DTranspose(n_classes, kernel_size=(2, 2), strides=(2, 2), padding='same')(u2_add)
    u4_add = Add()([p3_n,u4])
    
    outputs = Conv2DTranspose(n_classes, kernel_size=(8, 8), strides=(8, 8), padding='same',
                        activation='softmax')(u4_add)
     
    model = Model(inputs=[inputs], outputs=[outputs])
    
    return model

def modeloUnet32(n_classes, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS):

    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

    #Contraction path
    c1 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
    #c1 = Dropout(0.1)(c1)
    c1 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)
    
    c2 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    #c2 = Dropout(0.1)(c2)
    c2 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)
     
    c3 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    #c3 = Dropout(0.2)(c3)
    c3 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)
     
    c4 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    #c4 = Dropout(0.2)(c4)
    c4 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)
     
    c5 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    #c5 = Dropout(0.3)(c5)
    c5 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
    
    #Expansive path 
    u6 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    #c6 = Dropout(0.2)(c6)
    c6 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
     
    u7 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    #c7 = Dropout(0.2)(c7)
    c7 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
     
    u8 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    #c8 = Dropout(0.1)(c8)
    c8 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
     
    u9 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    #c9 = Dropout(0.1)(c9)
    c9 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
     
    outputs = Conv2D(n_classes, (1, 1), activation='softmax')(c9)
     
    model = Model(inputs=[inputs], outputs=[outputs])
    
    return model


#Criar Modelo
model = modeloUnet32(N_CLASSES,SIZE_X,SIZE_Y,N_CHANNEL)
model = modeloVgg16(N_CLASSES,SIZE_X,SIZE_Y,N_CHANNEL)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', tf.keras.metrics.MeanIoU(num_classes=N_CLASSES)])
model.summary()


#Carregar Modelo
model = keras.models.load_model('modeloUnetCSDAug1v9256x256Generator2.hdf5')
model = keras.models.load_model('modeloFCN Vgg16 8s CSDAug256x256.hdf5')
model = keras.models.load_model('UnetCSDAug 256x256G 7M.hdf5')
#Carregar video:
cap= cv2.VideoCapture('cruzamento 4.mp4')

#nome do video de saida:
nomeVideoSaida='Monografia\\videos\\cruzamento 4 processado.mp4'



#Numero de frames
numeroFrames=cap.get(cv2.CAP_PROP_FRAME_COUNT)

#Frames por Segundo
fps=cap.get(cv2.CAP_PROP_FPS)

#Tempo do video(topo)
tempoVideo=math.ceil(numeroFrames/fps)


#Gerar Frames
videoFrames=[]
i=0
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == False:
        break
    videoFrames.append(cv2.resize(frame,(SIZE_X,SIZE_Y)))
    ##cv2.imwrite('video_frames\\batida 2\\'+str(i)+'.png',cv2.resize(frame,(SIZE_X,SIZE_Y)))
    i+=1
cap.release()
print (str(len(videoFrames))+" frames gerados!")
del cap


#Processar
videoprocessado=[]
for i in tqdm(range(len(videoFrames))):
    novaImg=videoFrames[i]
    novaImg= cv2.cvtColor(novaImg, cv2.COLOR_BGR2GRAY)
    novaImg=np.array(novaImg)
    novaImg=np.expand_dims(novaImg, 0)
    novaImg= normalize(novaImg, axis=1)
    prediction = (model.predict(novaImg))
    predicted_img=np.argmax(prediction, axis=3)[0,:,:]
    videoprocessado.append(predicted_img)
videoprocessado=np.array(videoprocessado)


#Gerar video
#out = cv2.VideoWriter(nomeVideoSaida,cv2.VideoWriter_fourcc(*'DIVX'), fps, (256,256))  #.avi
out = cv2.VideoWriter(nomeVideoSaida,cv2.VideoWriter_fourcc(*'DIVX'), fps, (SIZE_X,SIZE_Y))  #.MP4V
for i in tqdm(range(len(videoprocessado))):
    videoExpandido=np.stack((videoprocessado[i],)*3, axis=-1)
    videoExpandido[np.all(videoExpandido == (1, 1, 1), axis=-1)] = (255,0,0) #pessoas
    videoExpandido[np.all(videoExpandido == (2, 2, 2), axis=-1)] = (255,140,0) #ciclista
    videoExpandido[np.all(videoExpandido == (3, 3, 3), axis=-1)] = (0,0,255) #carros
    #videoExpandido[np.all(videoExpandido == (4, 4, 4), axis=-1)] = (0,191,255) #caminhao
    videoExpandido[np.all(videoExpandido == (4, 4, 4), axis=-1)] = (106,82,152) #caminhao
    videoExpandido[np.all(videoExpandido == (5, 5, 5), axis=-1)] = (135,206,250) #onibus
    videoExpandido[np.all(videoExpandido == (6, 6, 6), axis=-1)] = (70,130,180) #vã/trailer
    videoExpandido[np.all(videoExpandido == (7, 7, 7), axis=-1)] = (112,128,144) #carreta
    videoExpandido[np.all(videoExpandido == (8, 8, 8), axis=-1)] = (0,255,255) #trem
    videoExpandido[np.all(videoExpandido == (9, 9, 9), axis=-1)] = (0,255,127) #moto 
    videoExpandido[np.all(videoExpandido == (10, 10, 10), axis=-1)] = (124,252,0) #biscicleta 
    img_blend = cv2.addWeighted(cv2.cvtColor(videoFrames[i], cv2.COLOR_RGB2BGR),0.4,videoExpandido,0.8,0,dtype = cv2.CV_8U)
    out.write(cv2.cvtColor(img_blend, cv2.COLOR_RGB2BGR))
out.release()

#Deleta variaveis da memoria
del videoprocessado
del videoFrames
