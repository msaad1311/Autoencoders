import numpy as np
from cv2 import cv2
import os
from numba import cuda
import matplotlib.pyplot as plt
import random
from skimage.metrics import structural_similarity as ssim
import image_slicer

from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Reshape, Input, InputLayer,Conv2D,MaxPool2D,UpSampling2D,Conv2DTranspose,Cropping2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.callbacks import ModelCheckpoint
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(gpus[0],
                                                                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=(1024*4))])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

def build_autoencoder(img_shape, code_size,name,shape_req):
    if name == 'Simple':
        # The encoder
        encoder = Sequential()
        encoder.add(InputLayer(img_shape))
        encoder.add(Flatten())
        encoder.add(Dense(code_size))

        # The decoder
        decoder = Sequential()
        decoder.add(InputLayer((code_size,)))
        decoder.add(Dense(np.prod(img_shape))) # np.prod(img_shape) is the same as 32*32*3, it's more generic than saying 3072
        decoder.add(Reshape(img_shape))

        return encoder, decoder
    
    elif name=='CNN2':
        encoder = Sequential()
        encoder.add(Conv2D(128,3,strides=2,input_shape=(shape_req,shape_req,3)))
        encoder.add(Conv2D(128,3,strides=2))
        encoder.add(Conv2D(128,3,strides=2))
        encoder.add(Conv2D(128,3,strides=1))
        encoder.add(Flatten())
        encoder.add(Dense(code_size))

        shape_pooling = encoder.layers[-3].output_shape
        shape_flatten = encoder.layers[-2].output_shape

        decoder = Sequential()
        decoder.add(Dense(shape_flatten[1],input_shape=(code_size,)))
        decoder.add(Reshape((shape_pooling[1],shape_pooling[2],shape_pooling[3])))
        decoder.add(Conv2DTranspose(128,3,strides=1))
        decoder.add(Conv2DTranspose(128,3,strides=2))
        decoder.add(Conv2DTranspose(128,3,strides=2))
        decoder.add(Conv2DTranspose(128,3,strides=2))
        decoder.add(Conv2DTranspose(3,3,strides=1))
        decoder.add(Cropping2D(cropping=((1, 0), (1, 0)), data_format=None))
        
        return encoder,decoder
    
    elif name == 'CNN':

        pad='valid'
        img_shape=(128,128,3)
        # code_size=512
        encoder = Sequential()
        encoder.add(Conv2D(filters=32,kernel_size=3,input_shape=(img_shape)))
        encoder.add(MaxPool2D(pool_size=(2,2)))
        encoder.add(Conv2D(filters=16,kernel_size=3))
        encoder.add(MaxPool2D(pool_size=(2,2)))
        encoder.add(Conv2D(filters=8,kernel_size=3))
        encoder.add(MaxPool2D(pool_size=(2,2)))
        encoder.add(Flatten())
        encoder.add(Dense(code_size))

        #The Decoder
        shape_pooling = encoder.layers[-3].output_shape
        shape_flatten = encoder.layers[-2].output_shape
        decoder = Sequential()
        decoder.add(Dense(shape_flatten[1],input_shape=(code_size,)))
        decoder.add(Reshape((shape_pooling[1],shape_pooling[2],shape_pooling[3])))
        decoder.add(UpSampling2D(size=(2,2)))
        decoder.add(Conv2DTranspose(filters=8,kernel_size=3,padding=pad))
        decoder.add(UpSampling2D(size=(2,2)))
        decoder.add(Conv2DTranspose(filters=16,kernel_size=3,padding=pad))
        decoder.add(UpSampling2D(size=(2,2)))
        decoder.add(Conv2DTranspose(filters=32,kernel_size=3,padding=pad))
        # decoder.add(UpSampling2D(size=(2,2)))
        decoder.add(Conv2DTranspose(filters=3,kernel_size=3,padding=pad))

        # pad='same'
        # #The encoder
        # encoder = Sequential()
        # encoder.add(Conv2D(filters=128,kernel_size=3,input_shape=(img_shape)))
        # encoder.add(MaxPool2D(pool_size=(2,2)))
        # encoder.add(Conv2D(filters=64,kernel_size=3))
        # encoder.add(MaxPool2D(pool_size=(2,2)))
        # encoder.add(Conv2D(filters=32,kernel_size=3))
        # encoder.add(MaxPool2D(pool_size=(2,2)))
        # encoder.add(Flatten())
        # encoder.add(Dense(code_size))

        # #The Decoder
        # shape_pooling = encoder.layers[-3].output_shape
        # shape_flatten = encoder.layers[-2].output_shape
        # decoder = Sequential()
        # decoder.add(Dense(shape_flatten[1],input_shape=(code_size,)))
        # decoder.add(Reshape((shape_pooling[1],shape_pooling[2],shape_pooling[3])))
        # decoder.add(UpSampling2D(size=(2,2)))
        # decoder.add(Conv2D(filters=32,kernel_size=3,padding=pad))
        # decoder.add(UpSampling2D(size=(2,2)))
        # decoder.add(Conv2D(filters=64,kernel_size=3,padding=pad))
        # decoder.add(UpSampling2D(size=(2,2)))
        # decoder.add(Conv2D(filters=128,kernel_size=3,padding=pad))
        # decoder.add(UpSampling2D(size=(2,2)))
        # decoder.add(Conv2D(filters=3,kernel_size=3,padding=pad))
        
        return encoder,decoder
    
def visualize(img,encoder,decoder):
    """Draws original, encoded and decoded images"""
    code = encoder.predict(img[None])[0]
    reco = decoder.predict(code[None])[0]

    plt.subplot(1,3,1)
    plt.title("Original")
    plt.imshow(np.clip(img, 0, 1))

#     plt.subplot(1,3,2)
#     plt.title("Code")
#     plt.imshow(code.reshape([code.shape[-1]//2,-1]))

    plt.subplot(1,3,3)
    plt.title("Reconstructed")
    plt.imshow(np.clip(reco, 0, 1))
    plt.show()
    
    print('-'*30,'SSIM:',ssim(img[0],reco[0],multichannel=True),'-'*30)
    
    return

### Metric
def overall_ssim(actual,predicted):
    assert len(actual)==len(predicted)
    results = []
    for i in range(len(actual)):
        s = ssim(actual[i],predicted[i],multichannel=True)
        results.append(s)
    results = np.array(results)
    return np.mean(results)

def read_video(path,video_name):
    
    path = path.replace('\\','/')
    cap = cv2.VideoCapture(video_name)
    fps=cap.get(cv2.CAP_PROP_FPS)
    width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH) # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) # float
    print('Loading Video...')
    index = 0
    successs,img = cap.read()

    path_old = os.getcwd()
    os.chdir(path)
    names=[]

    while successs:
        cv2.imwrite('frame_{}.jpg'.format(index),img)
        names.append('frame_{}.jpg'.format(index))
        successs,img = cap.read()
        index+=1
        
    os.chdir(path_old)
    
    print('Completed')
    
    return names,fps,int(width),int(height)

def read_imgs(path,names,slices,width,height):
    path = path.replace('\\','/')
    path_old=os.getcwd()
    os.chdir(path)
    img =[]
    temp_array =[]
    for name in names:
        temp = cv2.imread(name)
        temp = cv2.resize(temp,(width,height))
        cv2.imwrite(name,temp)
        temp = image_slicer.slice(name,slices,save=False)
        for s in range(slices):
            temp_array.append(np.asarray(temp[s].image))
    img = np.array(temp_array)
        # img.append(cv2.resize(temp,(width,height)))
    os.chdir(path_old)
    return img

def splitter(x,testsize):
    x_train,x_test = train_test_split(x,test_size=testsize,random_state=42)
    return x_train,x_test

def build_model(images,embedding_size,types,shape_req):
    IMG_SHAPE = images.shape[1:]
    embed_size = 1000
    encoder, decoder = build_autoencoder(IMG_SHAPE, embed_size,types,shape_req)

    inp = Input(IMG_SHAPE)
    code = encoder(inp)
    reconstruction = decoder(code)
    
    
    autoencoder = Model(inp,reconstruction)
    autoencoder.compile(optimizer='adam', loss='mse',metrics=['mae'])

    print('[INFO] The model is created')
    
    return autoencoder,encoder,decoder
    
    
def model_fit(model,x_train,x_test,epoch,batch_size=None):
    filepath = 'weights.hdf5'
    checkpoint = ModelCheckpoint(filepath,monitor='val_loss',verbose=0,save_best_only=True,mode=min)
    
    history = model.fit(x=x_train, y=x_train, epochs=epoch,validation_split=0.2,batch_size=batch_size,callbacks=[checkpoint])
    
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    
    return

    