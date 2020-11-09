## Libraries
import numpy as np
# from cv2 import cv2
import cv2
import os
import random
import sys
from tqdm import tqdm
import gc

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

#For Adrastea
# DATASET = r'C:\Users\Saad.LAKES\Desktop\Autoencoders\Videos\Dataset' ## where there is dataset
# DATASET_NAME = r'bunny_video.mp4' ## Name of the video
# FRAMES = r'C:\Users\Saad.LAKES\Desktop\Autoencoders\Videos\Results\Video_Frames' ## where you want to save the frames
# RECONSTRUCTED = r'C:\Users\Saad.LAKES\Desktop\Autoencoders\Videos\Results\ReconVideo_Frames' ## where the reconstructed frames are saved

#For Naon
DATASET = r'C:\Users\saad\Desktop\Autoencoders\Videos\Dataset' ## where there is dataset
DATASET_NAME = r'bunny_video.mp4' ## Name of the video
FRAMES = r'C:\Users\saad\Desktop\Autoencoders\Videos\Results\Video_Frames' ## where you want to save the frames
RECONSTRUCTED = r'C:\Users\saad\Desktop\Autoencoders\Videos\Results\ReconVideo_Frames' ## where the reconstructed frames are saved
WEIGHTS = r'C:\Users\saad\Desktop\Autoencoders\Videos\Results\Weights'
VIDEO = r'C:\Users\saad\Desktop\Autoencoders\Videos\Results\Video'

WIDTH = 640 ## width of the reconstrcted image
HEIGHT = 480 ## height of the reconstrcted image
FPS = 25 ## fps of the reconstrcted image
SLICES = 2 ## Number of slices to be made on the image

TEST_SIZE = 0.1
EMBEDDING_SIZE = 512 ## bottleneck layer nodes

## MAIN

#Loading file
names,fps,width,height = read_video(FRAMES,os.path.join(DATASET,DATASET_NAME))
print(f'The fps is {fps}, The width of the frame is {width} and the height of the frame is {height}')

#Scaling Images
images = np.array(read_imgs(FRAMES,names,SLICES,WIDTH,HEIGHT))
images = images.astype('float32')/255.
print(f'The shape of the images is {images.shape}')

#Sanity check on the images
ran = random.sample(range(0,len(images)),5)
for r in ran:
    plt.imshow(images[r])
    plt.show()
    
#Taking User Input
user = input('Do you want to continue? [Y/N]')
if user=='N':
    print('Shutting the Program Down.')
    sys.exit()

else:
    #Splitting the data in training and testing
    train,test = splitter(images,TEST_SIZE)
    print(f'The shape of train is {train.shape}')
    print(f'The shape of test is {test.shape}')
    
    #Creating the models
    img_shape = images.shape[1:]
    model1,model2 = build_autoencoder(img_shape,EMBEDDING_SIZE,'Hi-Res',None)
    
    model1.compile(loss='mse',optimizer='adam',metrics=['mae'])
    model2.compile(loss='mse',optimizer='adam',metrics=['mae'])
    
    #Fitting the first model
    model1 = model_fit(model1,'model1',WEIGHTS,train,train,200,5)
    
    #Visualizing the model results
    visualize(model1,train,5)
    
    #Preparing the data for second model
    print('-'*30,'Preparing dataset for the second model','-'*30)
    train_intermediate = data_prep(train,model1,None)
    
    #Sanity check for the shape
    assert train.shape == train_intermediate.shape
    
    #Fitting the second model
    model2 = model_fit(model2,'model2',WEIGHTS,train_intermediate,train,200,5)
    
    #Visualizing the output of second model
    visualize(model2,train_intermediate,5)
    
    predictions = data_prep(images,model1,model2)
    predictions = predictions * 255.
    
    print(f'The overall similarity index is : {overall_ssim(images,predictions)}')

    user_vid = input('Do You want to Generate Video? [Y/N]')
    if user_vid =='N':
        print('Exiting the program')
        sys.exit()
    else:
        video_creater('Recon.avi',VIDEO,25.,640,480,predictions)
        original_video('Original.avi',VIDEO,FRAMES,25.,names)
    
    
    
'''    
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
os.chdir(path_old).



'''