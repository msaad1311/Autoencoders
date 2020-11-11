import numpy as np
import cv2
import os
import random
from tqdm import tqdm

from Autoencoder import *
from make_video import *


def main():
    # PATHS
    DATASET = r'C:\Users\saad\Desktop\Autoencoders\Videos\Dataset'  # where there is dataset
    DATASET_NAME = ['bunny_video.mp4','Rabbit_video.mp4','bicycle_sequence.mp4'] # Name of the video
    # where you want to save the frames
    FRAMES = r'C:\Users\saad\Desktop\Autoencoders\Videos\Results'
    FRAME_NAME = ['bunny','rabbit','bicycle']
    FRAME_TRAIN =['bunny_train','rabbit_train','bicycle_train']
    FRAME_TEST = ['bunny_test','rabbit_test','bicycle_test']

    #Creating the video frames for each video
    for idx,f in enumerate(FRAME_NAME):
        names,fps,width,height = read_video(os.path.join(FRAMES,f),os.path.join(DATASET,DATASET_NAME[idx]))
        print(f'{f} completed')
    
    #Randomly selecting certain number of images
    
    
    
    return

if __name__ == "__main__":
    main()