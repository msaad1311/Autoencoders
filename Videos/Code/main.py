## Libraries
import numpy as np
# from cv2 import cv2
import cv2
import os
import random

from Autoencoder import *
from make_video import *

import tensorflow as tf

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
        
## PARAMETERS
PATH =  r'C:\Users\saad\Desktop\Autoencoders\Videos\Results\Video_Frames'
PATH_RESULTS = r'C:\Users\saad\Desktop\Autoencoders\Videos\Results\ReconVideo_Frames'
EMBEDDING_SIZE = 1000
SHAPE_REQUIRED = 256

## MAIN
names = read_video(PATH,r'C:\Users\saad\Desktop\Autoencoders\Videos\Dataset\bunny_video.mp4')
images = np.array(read_imgs(PATH,names,SHAPE_REQUIRED))
images = images.astype('float')/255.
print(images.shape)
train,test = splitter(images,0.1)
img_shape = np.prod(images.shape[1:])

print('[INFO] The preprocessing is complete. Moving to model creation')

model,encoder,decoder = build_model(images,EMBEDDING_SIZE,'CNN2',SHAPE_REQUIRED)
model.summary()

# model_fit(model,train,test,200,5)

model.load_weights('weights.hdf5')

# num_test = random.sample(range(0,len(test)),1)
# for i in num_test:
#     img = test[i]
#     visualize(img,encoder,decoder)

bottle = encoder.predict(test)
predicted = decoder.predict(bottle)

print(f'The average Similarity image is : {round(overall_ssim(test,predicted),2)}')
print(f'The compression rate is: {round((np.prod(img_shape)/EMBEDDING_SIZE),2)}')

path = PATH_RESULTS.replace('\\','/')
path_old = os.getcwd()
os.chdir(path)

out = cv2.VideoWriter('Reconstructed.avi',cv2.VideoWriter_fourcc(*"MJPG"),25.0,(256,256))
for i in range(len(images)):
    temp = np.expand_dims(images[i],axis=0)
    preds = decoder.predict(encoder.predict(temp))
    preds = preds.reshape(preds.shape[1],preds.shape[2],preds.shape[3])
    preds = preds * 255.
    # print(preds.shape)
    cv2.imwrite('frame_{}.jpg'.format(i),preds)
    out.write(preds)
out.release
os.chdir(path_old)