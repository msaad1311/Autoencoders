from tensorflow.keras.layers import Dense, Flatten, Reshape, Input, InputLayer,Conv2D,MaxPool2D,UpSampling2D
from tensorflow.keras.models import Sequential, Model
from skimage.metrics import structural_similarity as ssim
import numpy as np
import matplotlib.pyplot as pyplot

def build_autoencoder(img_shape, code_size,name):
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
    
    elif name == 'CNN':
        pad='same'
        #The encoder
        encoder = Sequential()
        encoder.add(Conv2D(filters=128,kernel_size=3,input_shape=(img_shape)))
        encoder.add(MaxPool2D(pool_size=(2,2)))
        encoder.add(Conv2D(filters=64,kernel_size=3))
        encoder.add(MaxPool2D(pool_size=(2,2)))
        encoder.add(Conv2D(filters=32,kernel_size=3))
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
        decoder.add(Conv2D(filters=32,kernel_size=3,padding=pad))
        decoder.add(UpSampling2D(size=(2,2)))
        decoder.add(Conv2D(filters=64,kernel_size=3,padding=pad))
        decoder.add(UpSampling2D(size=(2,2)))
        decoder.add(Conv2D(filters=128,kernel_size=3,padding=pad))
        decoder.add(UpSampling2D(size=(2,2)))
        decoder.add(Conv2D(filters=3,kernel_size=3,padding=pad))
        
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
