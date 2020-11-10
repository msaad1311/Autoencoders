import numpy as np
# from cv2 import cv2
import cv2
import os
from numba import cuda
import matplotlib.pyplot as plt
import random
from skimage.metrics import structural_similarity as ssim
import image_slicer
import gc
from tqdm import tqdm
import shutil

from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Reshape, Input, \
    InputLayer,Conv2D,MaxPool2D,UpSampling2D,Conv2DTranspose,Cropping2D,add
from tensorflow.keras.regularizers import l1
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
        if slices>=2:
            temp = image_slicer.slice(name,slices,save=False)
            for s in range(slices):
                temp_array.append(np.asarray(temp[s].image))
        else:
            temp_array.append(cv2.imread(name))
    img = np.array(temp_array)
    os.chdir(path_old)
    return img

def splitter(x,testsize):
    x_train,x_test = train_test_split(x,test_size=testsize,random_state=42)
    return x_train,x_test


def build_autoencoder(img_shape, code_size,name,shape_req):
    if name == 'Simple':
        # The encoder
        encoder = Sequential()
        encoder.add(InputLayer(img_shape))
        encoder.add(Dense(1,name='EDense1'))
        encoder.add(Reshape((encoder.get_layer('EDense1').output_shape[1],-1)))
        encoder.add(Dense(1,name='EDense2'))
        encoder.add(Flatten(name='EFlatten1'))
        encoder.add(Dense(code_size))

        # The decoder
        decoder = Sequential()
        decoder.add(InputLayer((code_size,)))
        decoder.add(Dense(encoder.get_layer('EFlatten1').output_shape[1]))
        decoder.add(Reshape((-1,1)))
        decoder.add(Dense(img_shape[1])) 
        decoder.add(Reshape(img_shape))

        return encoder, decoder
    
    elif name=='CNN':
        #The Encoder
        encoder = Sequential()
        encoder.add(InputLayer(img_shape))
        encoder.add(Conv2D(filters=32,kernel_size=3,strides=2,padding='same'))
        encoder.add(Conv2D(filters=32,kernel_size=3,strides=2,padding='same'))
        encoder.add(Conv2D(filters=32,kernel_size=3,strides=2,padding='same'))
        encoder.add(Conv2D(filters=32,kernel_size=3,strides=2,padding='same'))

        #The Decoder
        decoder = Sequential()
        decoder.add(InputLayer(encoder.output_shape[1:]))
        decoder.add(Conv2DTranspose(filters=32,kernel_size=3,strides=2,padding='same'))
        decoder.add(Conv2DTranspose(filters=32,kernel_size=3,strides=2,padding='same'))
        decoder.add(Conv2DTranspose(filters=32,kernel_size=3,strides=2,padding='same'))
        decoder.add(Conv2DTranspose(filters=3,kernel_size=3,strides=2,padding='same'))
        
        return encoder,decoder
    
    elif name=='Hi-Res':
        ##################################
        ####### First Autoencoder ########
        ##################################

        # The encoder 1
        encoder1_1 = Input(img_shape,name='encoder1_input')
        encoder1_2 = Conv2D(filters=32,kernel_size=3,strides=2,padding='same',name='E1Layer1')(encoder1_1)
        encoder1_3 = Conv2D(filters=32,kernel_size=3,strides=2,padding='same',name='E1Layer2')(encoder1_2)
        encoder1_4 = Conv2D(filters=32,kernel_size=3,strides=2,padding='same',name='E1Layer3')(encoder1_3)
        encoder1_5 = Conv2D(filters=32,kernel_size=3,strides=2,padding='same',name='E1Layer4')(encoder1_4)

        encoder1 = Model(encoder1_1,encoder1_5)

        # The decoder 1
        decoder1_1 = Input(encoder1.get_layer('E1Layer4').output_shape[1:],name='decoder1_input')
        decoder1_2 = Conv2DTranspose(filters=32,kernel_size=3,strides=2,padding='same',name='D1Layer1')(decoder1_1)
        decoder1_3 = Conv2DTranspose(filters=32,kernel_size=3,strides=2,padding='same',name='D1Layer2')(decoder1_2)
        decoder1_4 = Conv2DTranspose(filters=32,kernel_size=3,strides=2,padding='same',name='D1Layer3')(decoder1_3)
        decoder1_5 = Conv2DTranspose(filters=3,kernel_size=3,strides=2,padding='same',name='D1Layer4')(decoder1_4)

        decoder1 = Model(decoder1_1,decoder1_5)

        inp = Input(img_shape)
        code = encoder1(inp)
        reconstruction = decoder1(code)
        autoencoder1 = Model(inp,reconstruction)

        ##################################
        ####### Second Autoencoder #######
        ##################################

        # The encoder 2
        encoder2_1 = Input(img_shape,name='encoder2_input')
        encoder2_2 = Conv2D(filters=256,kernel_size=3,strides=2,padding='same',activation='relu',activity_regularizer=l1(10e-10),name='E2Layer1')(encoder2_1)
        encoder2_3 = Conv2D(filters=128,kernel_size=3,strides=2,padding='same',activation='relu',activity_regularizer=l1(10e-10),name='E2Layer2')(encoder2_2)
        encoder2_4 = Conv2D(filters=64,kernel_size=3,strides=2,padding='same',activation='relu',activity_regularizer=l1(10e-10),name='E2Layer3')(encoder2_3)
        encoder2_5 = Conv2D(filters=32,kernel_size=3,strides=2,padding='same',activation='relu',activity_regularizer=l1(10e-10),name='E2Layer4')(encoder2_4)

        # The decoder 2
        decoder2_2 = Conv2DTranspose(filters=64,kernel_size=3,strides=2,padding='same',activation='relu',activity_regularizer=l1(10e-10),name='D2Layer1')(encoder2_5)
        add2_1 = add([decoder2_2,encoder2_4]) # Residual Connection 
        decoder2_3 = Conv2DTranspose(filters=128,kernel_size=3,strides=2,padding='same',activation='relu',activity_regularizer=l1(10e-10),name='D2Layer2')(add2_1)
        decoder2_4 = Conv2DTranspose(filters=256,kernel_size=3,strides=2,padding='same',activation='relu',activity_regularizer=l1(10e-10),name='D2Layer3')(decoder2_3)
        add2_2 = add([decoder2_4,encoder2_2]) # Residual Connection 
        decoder2_5 = Conv2DTranspose(filters=3,kernel_size=3,strides=2,padding='same',activation='relu',activity_regularizer=l1(10e-10),name='D2Layer4')(add2_2)

        autoencoder2 = Model(encoder2_1,decoder2_5)
        
        return autoencoder1,autoencoder2

def model_fit(model,file_name,saving_path,X,Y,epoch,batch_size=None):

    path_old = os.getcwd()
    saving_path = saving_path.replace('\\','/')
    os.chdir(saving_path)

    filepath = '{}.hdf5'.format(file_name)
    checkpoint = ModelCheckpoint(filepath,monitor='val_loss',save_best_only=True)
    
    history = model.fit(x=X, y=Y, epochs=epoch,validation_split=0.2,batch_size=batch_size,callbacks=[checkpoint])
    
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    
    print('-'*30,'Loading the best weights','-'*30)
    model.load_weights(filepath)
    
    os.chdir(path_old)

    return model
        
    
def visualize(model,data,num_imgs):
    """Draws original, encoded and decoded images"""
    num = random.sample(range(0,len(data)),5)
    for i in num:
        im = data[i]
        pred=model.predict(np.expand_dims(im,axis=0))
        plt.subplot(121)
        plt.imshow(im)
        plt.title('Original')
        
        plt.subplot(122)
        plt.imshow(pred[0,:,:,:])
        plt.title('Reconstructed')
        
        plt.show()
        
        print('-'*30,overall_ssim(data[i],pred[0,:,:,:]))
    return

### Metric
def overall_ssim(actual,predicted):
    assert len(actual)==len(predicted)
    results = []
    for i in tqdm(range(len(actual)),desc='SSIM Calculation'):
        s = ssim(actual[i],predicted[i],multichannel=True)
        results.append(s)
    results = np.array(results)
    return np.mean(results)

def data_prep(data,model1,model2=None):
    temp = []
    for i in tqdm(range(len(data))):
        imgs_new = np.expand_dims(data[i],axis=0)
        if model2==None:
            temp.append(model1.predict(imgs_new))
        else:
            temp.append(model2.predict(model1.predict(imgs_new)))
        
        del(imgs_new)
        gc.collect()
    temp=np.array(temp)
    temp=temp.reshape(temp.shape[0],temp.shape[2],temp.shape[3],temp.shape[4])
        
    print(f'The shape of the input for the model is {temp.shape}')
    
    return temp

def cleanup(lst,path):
    path = path.replace('\\','/')
    for l in lst:
        try:
            shutil.rmtree(os.path.join(path,l))
            print(f'Removed {l}')
        except FileNotFoundError:
            print(f'{l} not found.')
            pass
        
    return

    


    
'''
The following code is not being used as it is really primitive and not
as good as the other ones
elif name=='CNN':
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

    #     pad='valid'
    #     img_shape=(128,128,3)
    #     # code_size=512
    #     encoder = Sequential()
    #     encoder.add(Conv2D(filters=32,kernel_size=3,input_shape=(img_shape)))
    #     encoder.add(MaxPool2D(pool_size=(2,2)))
    #     encoder.add(Conv2D(filters=16,kernel_size=3))
    #     encoder.add(MaxPool2D(pool_size=(2,2)))
    #     encoder.add(Conv2D(filters=8,kernel_size=3))
    #     encoder.add(MaxPool2D(pool_size=(2,2)))
    #     encoder.add(Flatten())
    #     encoder.add(Dense(code_size))

    #     #The Decoder
    #     shape_pooling = encoder.layers[-3].output_shape
    #     shape_flatten = encoder.layers[-2].output_shape
    #     decoder = Sequential()
    #     decoder.add(Dense(shape_flatten[1],input_shape=(code_size,)))
    #     decoder.add(Reshape((shape_pooling[1],shape_pooling[2],shape_pooling[3])))
    #     decoder.add(UpSampling2D(size=(2,2)))
    #     decoder.add(Conv2DTranspose(filters=8,kernel_size=3,padding=pad))
    #     decoder.add(UpSampling2D(size=(2,2)))
    #     decoder.add(Conv2DTranspose(filters=16,kernel_size=3,padding=pad))
    #     decoder.add(UpSampling2D(size=(2,2)))
    #     decoder.add(Conv2DTranspose(filters=32,kernel_size=3,padding=pad))
    #     # decoder.add(UpSampling2D(size=(2,2)))
    #     decoder.add(Conv2DTranspose(filters=3,kernel_size=3,padding=pad))

    #     # pad='same'
    #     # #The encoder
    #     # encoder = Sequential()
    #     # encoder.add(Conv2D(filters=128,kernel_size=3,input_shape=(img_shape)))
    #     # encoder.add(MaxPool2D(pool_size=(2,2)))
    #     # encoder.add(Conv2D(filters=64,kernel_size=3))
    #     # encoder.add(MaxPool2D(pool_size=(2,2)))
    #     # encoder.add(Conv2D(filters=32,kernel_size=3))
    #     # encoder.add(MaxPool2D(pool_size=(2,2)))
    #     # encoder.add(Flatten())
    #     # encoder.add(Dense(code_size))

    #     # #The Decoder
    #     # shape_pooling = encoder.layers[-3].output_shape
    #     # shape_flatten = encoder.layers[-2].output_shape
    #     # decoder = Sequential()
    #     # decoder.add(Dense(shape_flatten[1],input_shape=(code_size,)))
    #     # decoder.add(Reshape((shape_pooling[1],shape_pooling[2],shape_pooling[3])))
    #     # decoder.add(UpSampling2D(size=(2,2)))
    #     # decoder.add(Conv2D(filters=32,kernel_size=3,padding=pad))
    #     # decoder.add(UpSampling2D(size=(2,2)))
    #     # decoder.add(Conv2D(filters=64,kernel_size=3,padding=pad))
    #     # decoder.add(UpSampling2D(size=(2,2)))
    #     # decoder.add(Conv2D(filters=128,kernel_size=3,padding=pad))
    #     # decoder.add(UpSampling2D(size=(2,2)))
    #     # decoder.add(Conv2D(filters=3,kernel_size=3,padding=pad))
        
    #     return encoder,decoder
    
    



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
    
    

'''