import os
import keras
import tensorflow as tf
from tensorflow.python.client import device_lib
import platform
from CustomDataGeneratorV2 import CustomDataGenerator as cdg
import numpy as np
import datetime
import modelsImplementation as m
from keras.utils import normalize
import tqdm
import cv2
import glob
import random
from matplotlib import pyplot as plt
import pickle
import time

SIZE_X=256
SIZE_Y=256
N_CHANNEL=1

#Quantidade de classes dos pixels
N_CLASSES=11 #classes: 0 a 10

#Informacões
print("Número de GPUs Disponiveis: ", len(tf.config.list_physical_devices('GPU')))
print("GPU: NVIDIA GeForce GTX 1070")
print("Versão Tensorflow: ",tf.__version__)
print("Versão Keras: ",keras.__version__)
print("Versão Python: ",platform.python_version())
print("Sistema Operacional: ",platform.system(), platform.release(), "- Versão: ",platform.version())

nomeModelos=['fcn16s-Aug_Final', 'fcn8s-Aug_Final', 'unet32-noAug_Final', 'unet32-Aug_Final','unet16-Aug_Final','fcn16s-Aug_Final2','fcn8s-Aug_Final2']
N_nome=4

#Instanciar gerador 256x256 sem augmentation
idList = os.listdir(r'cityscapesdataset\csdataset\train_images_new\img')

training_generator = cdg(idList, batch_size=16,size_x=SIZE_X, size_y=SIZE_Y, n_channels=N_CHANNEL, n_classes=N_CLASSES, shuffle=True, 
                         path_images='cityscapesdataset\\csdataset\\train_images_new\\img\\',
                         path_masks='cityscapesdataset\\csdataset\\train_masks_new\\img\\')

idListVal = os.listdir(r'cityscapesdataset\csdataset\val_images_new\img')

validation_generator = cdg(idListVal, batch_size=16,size_x=SIZE_X, size_y=SIZE_Y, n_channels=N_CHANNEL, n_classes=N_CLASSES, shuffle=True, 
                         path_images='cityscapesdataset\\csdataset\\val_images_new\\img\\',
                         path_masks='cityscapesdataset\\csdataset\\val_masks_new\\img\\')

#Instanciar gerador 256x256 com augmentation
idList = os.listdir(r'cityscapesdataset\csdataset\train_images_new_augmentation\img')

training_generator = cdg(idList, batch_size=16,size_x=SIZE_X, size_y=SIZE_Y, n_channels=N_CHANNEL, n_classes=N_CLASSES, shuffle=True, 
                         path_images='cityscapesdataset\\csdataset\\train_images_new_augmentation\\img\\',
                         path_masks='cityscapesdataset\\csdataset\\train_masks_new_augmentation\\img\\')

idListVal = os.listdir(r'cityscapesdataset\csdataset\val_images_new\img')

validation_generator = cdg(idListVal, batch_size=16,size_x=SIZE_X, size_y=SIZE_Y, n_channels=N_CHANNEL, n_classes=N_CLASSES, shuffle=True, 
                         path_images='cityscapesdataset\\csdataset\\val_images_new\\img\\',
                         path_masks='cityscapesdataset\\csdataset\\val_masks_new\\img\\')

#Instanciar gerador 128x128
idList = os.listdir(r'cityscapesdataset\csdatasetcp\train_images\img')

training_generator = cdg(idList, batch_size=8,size_x=SIZE_X, size_y=SIZE_Y, n_channels=N_CHANNEL, n_classes=N_CLASSES, shuffle=True, 
                         path_images='cityscapesdataset\\csdatasetcp\\train_images\\img\\',
                         path_masks='cityscapesdataset\\csdatasetcp\\train_masks\\img\\')

idListVal = os.listdir(r'cityscapesdataset\csdatasetcp\val_images\img')

validation_generator = cdg(idListVal, batch_size=8,size_x=SIZE_X, size_y=SIZE_Y, n_channels=N_CHANNEL, n_classes=N_CLASSES, shuffle=True, 
                         path_images='cityscapesdataset\\csdatasetcp\\val_images\\img\\',
                         path_masks='cityscapesdataset\\csdatasetcp\\val_masks\\img\\')

#Instanciar gerador 512x512
idList = os.listdir(r'cityscapesdataset\csdataset512\train_images\img')

training_generator = cdg(idList, batch_size=8,size_x=SIZE_X, size_y=SIZE_Y, n_channels=N_CHANNEL, n_classes=N_CLASSES, shuffle=True, 
                         path_images='cityscapesdataset\\csdataset512\\train_images\\img\\',
                         path_masks='cityscapesdataset\\csdataset512\\train_masks\\img\\')

idListVal = os.listdir(r'cityscapesdataset\csdataset512\val_images\img')

validation_generator = cdg(idListVal, batch_size=8,size_x=SIZE_X, size_y=SIZE_Y, n_channels=N_CHANNEL, n_classes=N_CLASSES, shuffle=True, 
                         path_images='cityscapesdataset\\csdataset512\\val_images\\img\\',
                         path_masks='cityscapesdataset\\csdataset512\\val_masks\\img\\')




#Criar Modelo
model = m.modeloFCN16s(N_CLASSES,SIZE_X,SIZE_Y,N_CHANNEL)
model = m.modeloFCN8s(N_CLASSES,SIZE_X,SIZE_Y,N_CHANNEL)
model = m.modeloFCN8s2(N_CLASSES,SIZE_X,SIZE_Y,N_CHANNEL)
model = m.modeloUnet32(N_CLASSES,SIZE_X,SIZE_Y,N_CHANNEL)
model = m.modeloUnet(N_CLASSES,SIZE_X,SIZE_Y,N_CHANNEL)
model = m.modeloUnet16(N_CLASSES,SIZE_X,SIZE_Y,N_CHANNEL)
model = m.modeloUnetBest(N_CLASSES,SIZE_X,SIZE_Y,N_CHANNEL)
model = m.unetk(N_CLASSES,SIZE_X,SIZE_Y,N_CHANNEL)

#opt = keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[tf.keras.metrics.MeanIoU(num_classes=N_CLASSES)])
model.summary()

#CallBacks
callback_ES = keras.callbacks.EarlyStopping(monitor='val_loss', patience=8)

checkpoint_filepath = 'checkpoints\\unet32_128-nodropout.hdf5'

checkpoint_filepath = 'checkpoints\\'+nomeModelos[N_nome]+'-bestWeightsVal_monografia.hdf5'
checkpoint_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath, save_weights_only=True, monitor='val_loss', mode='min', save_best_only=True)

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs/"+nomeModelos[N_nome]+"-"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S"), histogram_freq=1)

class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)

time_callback = TimeHistory()

model.fit(training_generator,epochs=50,verbose=1,callbacks=[callback_ES, checkpoint_callback, tensorboard_callback, time_callback],validation_data=validation_generator)

hist=model.fit(training_generator,epochs=50,verbose=1,callbacks=[checkpoint_callback, callback_ES, time_callback],validation_data=validation_generator)

times = time_callback.times
print("Tempo: ",sum(times)/60)

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

plt.plot(hist.history['mean_io_u'])
plt.plot(hist.history['val_mean_io_u'])
plt.title('mean iou')
plt.ylabel('mean iou')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

#Salvar History
with open('logs train\\fcn8s-Aug_FinalMonografia2', 'wb') as file_pi:
        pickle.dump(hist.history, file_pi)
        
with open('checkpoints\\times\\fcn8s-Aug_FinalMonografia2', 'wb') as file_pi:
        pickle.dump(times, file_pi)
        
hist = pickle.load(open('logs train\\unet16-256_nodropout', "rb"))

history2 = pickle.load(open('checkpoints\\times\\unet16-512x512_5', "rb"))

model.save(nomeModelos[N_nome]+'.hdf5')
model.save('unet32-128-nodropout_best.hdf5')
model = keras.models.load_model(nomeModelos[N_nome]+'.hdf5')
model = keras.models.load_model('unet32-Aug_Final.hdf5')
model.load_weights('checkpoints\\'+nomeModelos[N_nome]+'-bestWeightsVal_monografia.hdf5')
model.load_weights('checkpoints\\unet32_128-nodropout.hdf5')

#Test generator 256x256
idListTest = os.listdir(r'cityscapesdataset\csdataset\test_images')

test_generator = cdg(idListTest, batch_size=16,size_x=SIZE_X, size_y=SIZE_Y, n_channels=N_CHANNEL, n_classes=N_CLASSES, shuffle=True, 
                         path_images='cityscapesdataset\\csdataset\\test_images\\',
                         path_masks='cityscapesdataset\\csdataset\\test_masks_transformed\\img\\')

#Test generator 128x128
idListTest = os.listdir(r'cityscapesdataset\csdatasetcp\test_images\img')

test_generator = cdg(idListTest, batch_size=8,size_x=SIZE_X, size_y=SIZE_Y, n_channels=N_CHANNEL, n_classes=N_CLASSES, shuffle=True, 
                         path_images='cityscapesdataset\\csdatasetcp\\test_images\\img\\',
                         path_masks='cityscapesdataset\\csdatasetcp\\test_masks\\img\\')

#Test generator 512x512
idListTest = os.listdir(r'cityscapesdataset\csdataset512\test_images\img')

test_generator = cdg(idListTest, batch_size=8,size_x=SIZE_X, size_y=SIZE_Y, n_channels=N_CHANNEL, n_classes=N_CLASSES, shuffle=True, 
                         path_images='cityscapesdataset\\csdataset512\\test_images\\img\\',
                         path_masks='cityscapesdataset\\csdataset512\\test_masks\\img\\')


model.evaluate(test_generator, verbose=1)


###############
DATA_DIR_TEST="cityscapesdataset\\csdataset\\test_images"
DATA_DIR_TEST_MASKS="cityscapesdataset\\csdataset\\test_masks_transformed\\img"

testImages=[]
for i in range(len(os.listdir(DATA_DIR_TEST))):
    novaImg=cv2.imread(glob.glob('cityscapesdataset\\csdataset\\test_images\\'+str(i)+'.png')[0],0)
    #rgbNovaImg = cv2.cvtColor(novaImg, cv2.COLOR_BGR2RGB)
    #testImages.append(rgbNovaImg)
    testImages.append(novaImg)
        
testImages = np.array(testImages)

testMasks=[]
for i in range(len(os.listdir(DATA_DIR_TEST_MASKS))):
    novaImg=cv2.imread(glob.glob('cityscapesdataset\\csdataset\\test_masks_transformed\\img\\'+str(i)+'.png')[0],0)
    testMasks.append(novaImg)
    
#Converter lista para numpy array 
testMasks = np.array(testMasks)
print("Classes:",np.unique(testMasks))
print(len(np.unique(testMasks)))

testImages=np.expand_dims(testImages, axis=3)
xTestNormalized=normalize(testImages, axis=1)

def testeRede():
    testImgNumber = random.randint(0, len(testImages)-1)
    testImgNormalized = xTestNormalized[testImgNumber]
    #ground_truth=testMasksCateg[testImgNumber]
    ground_truth=testMasks[testImgNumber]
    testImgInput=np.expand_dims(testImgNormalized, 0)
    prediction = (model.predict(testImgInput))
    predicted_img=np.argmax(prediction, axis=-1)[0,:,:]

    print("Numero da Imagem:",testImgNumber)
    print("Shape ground truth:",ground_truth.shape)
    print("Shape prediction:",prediction[0].shape)
    print("Shape predicted_img:",predicted_img.shape)
    
    plt.figure(figsize=(16, 10))
    plt.subplot(131)
    plt.title('Imagem de teste normalizada')
    plt.imshow(testImages[testImgNumber])
    plt.subplot(132)
    plt.title('Label da imagem de teste')
    #plt.imshow(ground_truth[:,:,0])
    plt.imshow(ground_truth)
    plt.subplot(133)
    plt.title('Predição na imagem de teste')
    plt.imshow(predicted_img)
    plt.show()

testeRede()
