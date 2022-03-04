import os
import cv2 
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import random
import glob
#import keras
#import tensorflow as tf
import skimage as sk
#from skimage import transform
#from skimage import util
from CustomDataGeneratorV2 import CustomDataGenerator as cdg
#from PIL import Image
##from sklearn.utils import class_weight
import collections

#load images
masks=[]
for i in tqdm(range(len(os.listdir(r'cityscapesdataset\csdataset\train_masks_new_augmentation\img')))):
    masks.append(cv2.imread(glob.glob('cityscapesdataset\\csdataset\\train_masks_new_augmentation\\img\\'+str(i)+'.png')[0],0))

masks = np.array(masks)

np.unique(masks)


images=[]
for i in tqdm(range(len(os.listdir(r'cityscapesdataset\csdataset\train_images_new\img')))):
    images.append(cv2.imread(glob.glob('cityscapesdataset\\csdataset\\train_images_new\\img\\'+str(i)+'.png')[0]))
    
images = np.array(images)

#load validation
masksVal=[]
for i in tqdm(range(len(os.listdir(r'cityscapesdataset\csdataset\val_masks_new\img')))):
    masksVal.append(cv2.imread(glob.glob('cityscapesdataset\\csdataset\\val_masks_new\\img\\'+str(i)+'.png')[0],0))
    
masksVal = np.array(masksVal)

np.unique(masksVal)


imagesVal=[]
for i in tqdm(range(len(os.listdir(r'cityscapesdataset\csdataset\val_images_new\img')))):
    imagesVal.append(cv2.imread(glob.glob('cityscapesdataset\\csdataset\\val_images_new\\img\\'+str(i)+'.png')[0]))
    
imagesVal = np.array(imagesVal)

#plot images
randNumber=4296
randNumber=random.randint(0,len(images))
print(randNumber)
plt.figure(figsize=(16, 10))
plt.subplot(121)
plt.imshow(images[randNumber])
plt.subplot(122)
plt.imshow(masks[randNumber])
plt.show()

#plot val
randNumber=random.randint(0,len(imagesVal))
print(randNumber)
plt.figure(figsize=(16, 10))
plt.subplot(121)
plt.imshow(imagesVal[randNumber])
plt.subplot(122)
plt.imshow(masksVal[randNumber])
plt.show()

p=8
randNumber=p
p+=1

np.unique(masks[randNumber])

for img in masks:
    if 4 in np.unique(img):
        p = img
        break

xmg=cv2.imread('cityscapesdataset\\csdataset\\train_images_aug\\img\\3991.png')
ymg=cv2.imread('cityscapesdataset\\csdataset\\train_masks_aug\\img\\3991.png',0)

plt.imshow(xmg)
plt.show()
plt.imshow(ymg)
#################### Generator
idList = os.listdir(r'cityscapesdataset\csdataset\train_images_aug\img')

training_generator = cdg(idList, batch_size=16,size_x=256, size_y=256, n_channels=1, n_classes=11, shuffle=True, 
                         path_images='cityscapesdataset\\csdataset\\train_images_aug\\img\\',
                         path_masks='cityscapesdataset\\csdataset\\train_masks_aug\\img\\')
####################################
idx = 2425
class1=[]
i=0
for img in masksVal:
    if 1 in np.unique(img):
        class1.append(i)
    i+=1
    
class2=[]
i=0
for img in masksVal:
    if 2 in np.unique(img):
        class2.append(i)
    i+=1
    
class3=[]
i=0
for img in masksVal:
    if 3 in np.unique(img):
        class3.append(i)
    i+=1
    
class4=[]
i=0
for img in masksVal:
    if 4 in np.unique(img):
        class4.append(i)
    i+=1
    
class5=[]
i=0
for img in masksVal:
    if 5 in np.unique(img):
        class5.append(i)
    i+=1 
    

class6=[]
i=0
for img in masksVal:
    if 6 in np.unique(img):
        class6.append(i)
    i+=1

class7=[]
i=0
for img in masksVal:
    if 7 in np.unique(img):
        class7.append(i)
    i+=1
    

class8=[]
i=0
for img in masksVal:
    if 8 in np.unique(img):
        class8.append(i)
    i+=1
  
class9=[]
i=0
for img in masksVal:
    if 9 in np.unique(img):
        class9.append(i)
    i+=1
    
class10=[]
i=0
for img in masksVal:
    if 10 in np.unique(img):
        class10.append(i)
    i+=1

# criar lista para mover
mover=class6[:9]

for i in range(13):
    mover.append(class7[i])
    
for i in range(15):
    mover.append(class8[i])
    
for i in range(26):
    mover.append(class5[i])
    
for i in range(41):
    mover.append(class4[i])
    
for i in range(53):
    mover.append(class9[i])
    
for i in range(91):
    mover.append(class2[i])
    
for i in range(65):
    mover.append(class10[i])
    
for i in range(200,264):
    mover.append(class10[i])
    
for i in range(15):
    mover.append(class1[i])

moverND = list(dict.fromkeys(mover))

moverND.append(9)

moverNP = sorted(moverND)

idx=2425
for i in moverNP:
    os.rename('cityscapesdataset\\csdataset\\val_images_new\\img\\'+str(i)+'.png','cityscapesdataset\\csdataset\\train_images_new\\img\\'+str(idx)+'.png')
    os.rename('cityscapesdataset\\csdataset\\val_masks_new\\img\\'+str(i)+'.png','cityscapesdataset\\csdataset\\train_masks_new\\img\\'+str(idx)+'.png')
    idx+=1
   
indexval=0
for i in os.listdir('cityscapesdataset\\csdataset\\val_images_new\\img'):
    os.rename('cityscapesdataset\\csdataset\\val_images_new\\img\\'+i,'cityscapesdataset\\csdataset\\val_images_new\\img2\\'+str(indexval)+'.png')
    os.rename('cityscapesdataset\\csdataset\\val_masks_new\\img\\'+i,'cityscapesdataset\\csdataset\\val_masks_new\\img2\\'+str(indexval)+'.png')
    indexval+=1

###################
count1=0
count2=0
count3=0
count4=0
count5=0
count6=0
count7=0
count8=0
count9=0
count10=0


for img in masks:
    unq=np.unique(img)
    if 1 in unq:
        count1+=1
    if 2 in unq:
        count2+=1
    if 3 in unq:
        count3+=1
    if 4 in unq:
        count4+=1
    if 5 in unq:
        count5+=1
    if 6 in unq:
        count6+=1
    if 7 in unq:
        count7+=1
    if 8 in unq:
        count8+=1
    if 9 in unq:
        count9+=1
    if 10 in unq:
        count10+=1
        
count1v=0
count2v=0
count3v=0
count4v=0
count5v=0
count6v=0
count7v=0
count8v=0
count9v=0
count10v=0

for img in masksVal:
    unq=np.unique(img)
    if 1 in unq:
        count1v+=1
    if 2 in unq:
        count2v+=1
    if 3 in unq:
        count3v+=1
    if 4 in unq:
        count4v+=1
    if 5 in unq:
        count5v+=1
    if 6 in unq:
        count6v+=1
    if 7 in unq:
        count7v+=1
    if 8 in unq:
        count8v+=1
    if 9 in unq:
        count9v+=1
    if 10 in unq:
        count10v+=1

# set width of bar
barWidth = 0.25
fig = plt.subplots(figsize =(16, 8))
 
# set height of bar
Train = [count1, count2, count3, count4, count5, count6, count7, count8, count9, count10]
Validation = [count1v, count2v, count3v, count4v, count5v, count6v, count7v, count8v, count9v, count10v]
 
# Set position of bar on X axis
br1 = np.arange(len(Train))
br2 = [x + barWidth for x in br1]
 
# Make the plot
plt.bar(br1, Train, color ='r', width = barWidth,
        edgecolor ='grey', label ='Train')
plt.bar(br2, Validation, color ='g', width = barWidth,
        edgecolor ='grey', label ='Validation')
 
# Adding Xticks
plt.xlabel('Classes', fontweight ='bold', fontsize = 15)
plt.ylabel('Ocorrencias', fontweight ='bold', fontsize = 15)
plt.xticks([r + barWidth for r in range(len(Train))],
        ['1', '2', '3', '4', '5', '6', '7', '8', '9','10'])
 
plt.legend()
plt.show()

############  Gerar lista de imagens por classe
class1=[]
i=0
for img in masks:
    if 1 in np.unique(img):
        class1.append(i)
    i+=1
    
class2=[]
i=0
for img in masks:
    if 2 in np.unique(img):
        class2.append(i)
    i+=1
    
class3=[]
i=0
for img in masks:
    if 3 in np.unique(img):
        class3.append(i)
    i+=1
    
class4=[]
i=0
for img in masks:
    if 4 in np.unique(img):
        class4.append(i)
    i+=1
    
class5=[]
i=0
for img in masks:
    if 5 in np.unique(img):
        class5.append(i)
    i+=1 
    

class6=[]
i=0
for img in masks:
    if 6 in np.unique(img):
        class6.append(i)
    i+=1

class7=[]
i=0
for img in masks:
    if 7 in np.unique(img):
        class7.append(i)
    i+=1
    

class8=[]
i=0
for img in masks:
    if 8 in np.unique(img):
        class8.append(i)
    i+=1
  
class9=[]
i=0
for img in masks:
    if 9 in np.unique(img):
        class9.append(i)
    i+=1
    
class10=[]
i=0
for img in masks:
    if 10 in np.unique(img):
        class10.append(i)
    i+=1
    

###########################################
class45= list(set(class4) | set(class5))
class456=list(set(class45) | set(class6))

class4567= list(set(class456) | set(class7))

class45678=list(set(class4567) | set(class8))

class456789= list(set(class45678) | set(class9))

x=sorted(class456789)
####################### FLIP HORIZONTAL
imageFlip = np.fliplr(images[69])
maskFlip = np.fliplr(masks[69])

plt.imshow(imageFlip)
plt.show()
plt.imshow(maskFlip)

idx=2675
#for t in tqdm(range(len(images))):
for t in tqdm(class456789):
    imgFlip=np.fliplr(images[t])
    maskFlip=np.fliplr(masks[t])
    cv2.imwrite('cityscapesdataset\\csdataset\\train_images_new_augmentation\\img\\'+str(idx)+'.png',imgFlip)
    cv2.imwrite('cityscapesdataset\\csdataset\\train_masks_new_augmentation\\img\\'+str(idx)+'.png',maskFlip)
    idx+=1
#max 4151
idx=3779
intersect=sorted(list(set.intersection(set(class1), set(x))))

class1MenosIntersect=[x for x in class1 if x not in intersect]
class1MenosIntersect=sorted(class1MenosIntersect)

for y in range(len(class1MenosIntersect)-500):
    imgFlip=np.fliplr(images[class1MenosIntersect[y]])
    maskFlip=np.fliplr(masks[class1MenosIntersect[y]])
    cv2.imwrite('cityscapesdataset\\csdataset\\train_images_new_augmentation\\img\\'+str(idx)+'.png',imgFlip)
    cv2.imwrite('cityscapesdataset\\csdataset\\train_masks_new_augmentation\\img\\'+str(idx)+'.png',maskFlip)
    idx+=1

####################### Rotacao 
random_degree = random.uniform(-25, 25)
rot=sk.transform.rotate(images[4295], random_degree, preserve_range=False)
rotm=sk.transform.rotate(masks[4295],random_degree, preserve_range=True)
plt.imshow(rot)
plt.show()
plt.imshow(rotm)

############################# Horizontal Shift
image = cv2.imread(r'cityscapesdataset\csdataset\train_images_augmentation\img\1.png')
mask = cv2.imread(r'cityscapesdataset\csdataset\train_masks_augmentation\img\1.png',0)
plt.imshow(image)
num_rows, num_cols = image.shape[:2]
translation_matrix = np.float32([ [1,0,70], [0,1,0] ])
img_translation = cv2.warpAffine(image, translation_matrix, (num_cols,num_rows))
mask_translation = cv2.warpAffine(mask, translation_matrix, (num_cols,num_rows))

idx=4497
imagensAlteradas=np.linspace(50,4000,num=1800,dtype=int)
for i in range(len(imagensAlteradas)):
    img=cv2.imread('cityscapesdataset\\csdataset\\train_images_new_augmentation\\img\\'+str(imagensAlteradas[i])+'.png')
    mask=cv2.imread('cityscapesdataset\\csdataset\\train_masks_new_augmentation\\img\\'+str(imagensAlteradas[i])+'.png',0)
    randEscolha=random.randint(0,2) #0: esquerda, #1: direita, #2: cima
    if randEscolha==0:
        randTranslation=random.randint(40,128)
        translation_matrix = np.float32([ [1,0,-randTranslation], [0,1,0] ])
        img_translation = cv2.warpAffine(img, translation_matrix, (num_cols,num_rows))
        mask_translation = cv2.warpAffine(mask, translation_matrix, (num_cols,num_rows))
        cv2.imwrite('cityscapesdataset\\csdataset\\train_images_new_augmentation\\img\\'+str(idx)+'.png',img_translation)
        cv2.imwrite('cityscapesdataset\\csdataset\\train_masks_new_augmentation\\img\\'+str(idx)+'.png',mask_translation)
    elif randEscolha==1:
        randTranslation=random.randint(40,128)
        translation_matrix = np.float32([ [1,0,randTranslation], [0,1,0] ])
        img_translation = cv2.warpAffine(img, translation_matrix, (num_cols,num_rows))
        mask_translation = cv2.warpAffine(mask, translation_matrix, (num_cols,num_rows))
        cv2.imwrite('cityscapesdataset\\csdataset\\train_images_new_augmentation\\img\\'+str(idx)+'.png',img_translation)
        cv2.imwrite('cityscapesdataset\\csdataset\\train_masks_new_augmentation\\img\\'+str(idx)+'.png',mask_translation)
    else:
        randTranslation=random.randint(40,100)
        translation_matrix = np.float32([ [1,0,0], [0,1,-randTranslation] ])
        img_translation = cv2.warpAffine(img, translation_matrix, (num_cols,num_rows))
        mask_translation = cv2.warpAffine(mask, translation_matrix, (num_cols,num_rows))
        cv2.imwrite('cityscapesdataset\\csdataset\\train_images_new_augmentation\\img\\'+str(idx)+'.png',img_translation)
        cv2.imwrite('cityscapesdataset\\csdataset\\train_masks_new_augmentation\\img\\'+str(idx)+'.png',mask_translation)
    idx+=1

##################Salt Pepper
def sp_noise(image,prob):
    '''
    Add salt and pepper noise to image
    prob: Probability of the noise
    '''
    output = np.zeros(image.shape,np.uint8)
    thres = 1 - prob 
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output

def sp_noise2(image, prob):
    '''
    Add salt and pepper noise to image
    prob: Probability of the noise
    '''
    output = image.copy()
    if len(image.shape) == 2:
        black = 0
        white = 255            
    else:
        colorspace = image.shape[2]
        if colorspace == 3:  # RGB
            black = np.array([0, 0, 0], dtype='uint8')
            white = np.array([255, 255, 255], dtype='uint8')
        else:  # RGBA
            black = np.array([0, 0, 0, 255], dtype='uint8')
            white = np.array([255, 255, 255, 255], dtype='uint8')
    probs = np.random.random(image.shape[:2])
    image[probs < (prob / 2)] = black
    image[probs > 1 - (prob / 2)] = white
    return image

image = cv2.imread(r'cityscapesdataset\csdataset\train_images_augmentation\img\1.png')
noise_img = sp_noise(image,0.05)
noise_img2 = sp_noise2(image,0.03)
plt.imshow(image)
plt.show()
plt.imshow(noise_img)
plt.show()
plt.imshow(noise_img2)

from keras.utils import normalize

t = normalize(noise_img2,axis=1)
x= normalize(image, axis=1)
#cv2.imwrite('sp_noise.jpg', noise_img)

cv2.imwrite('cityscapesdataset\\csdataset\\train_masks_aug2\\1.png',noise_img2)


imagensAlteradas=np.linspace(0,6100,num=1800,dtype=int)
idx=6297
for i in range(len(imagensAlteradas)):
    img = cv2.imread('cityscapesdataset\\csdataset\\train_images_new_augmentation\\img\\'+str(imagensAlteradas[i])+'.png')
    mask = cv2.imread('cityscapesdataset\\csdataset\\train_masks_new_augmentation\\img\\'+str(imagensAlteradas[i])+'.png',0)
    randEscolha=random.randint(0,1)
    if randEscolha == 0:
        noise_img=sp_noise2(img, 0.03)
    else:
        noise_img=sp_noise2(img, 0.05)
    cv2.imwrite('cityscapesdataset\\csdataset\\train_images_new_augmentation\\img\\'+str(idx)+'.png',noise_img)
    cv2.imwrite('cityscapesdataset\\csdataset\\train_masks_new_augmentation\\img\\'+str(idx)+'.png',mask)
    idx+=1

###################### Criar validation Set
#550 imagens
valImages= np.linspace(10,2935,num=550,dtype=int)

for i in range(len(valImages)):
    os.rename('cityscapesdataset\\csdataset\\train_images_new\\img\\'+str(valImages[i])+'.png','cityscapesdataset\\csdataset\\val_images_new\\img\\'+str(valImages[i])+'.png')
    os.rename('cityscapesdataset\\csdataset\\train_masks_new\\img\\'+str(valImages[i])+'.png','cityscapesdataset\\csdataset\\val_masks_new\\img\\'+str(valImages[i])+'.png')

listDirTrain = os.listdir(r'cityscapesdataset\csdataset\train_images_new\img')

idx=0
for i in range(len(listDirTrain)):
    os.rename('cityscapesdataset\\csdataset\\train_images_new\\img\\'+listDirTrain[i],'cityscapesdataset\\csdataset\\train_images_new\\img2\\'+str(idx)+'.png')
    os.rename('cityscapesdataset\\csdataset\\train_masks_new\\img\\'+listDirTrain[i],'cityscapesdataset\\csdataset\\train_masks_new\\img2\\'+str(idx)+'.png')
    idx+=1
    
listDirTrain = os.listdir(r'cityscapesdataset\csdataset\val_images_new\img')

idx=0
for i in range(len(listDirTrain)):
    os.rename('cityscapesdataset\\csdataset\\val_images_new\\img\\'+listDirTrain[i],'cityscapesdataset\\csdataset\\val_images_new\\img2\\'+str(idx)+'.png')
    os.rename('cityscapesdataset\\csdataset\\val_masks_new\\img\\'+listDirTrain[i],'cityscapesdataset\\csdataset\\val_masks_new\\img2\\'+str(idx)+'.png')
    idx+=1

######### Compute Class weights
masks_reshaped = masks.reshape(-1,)

class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(masks_reshaped),
                                                 masks_reshaped)

