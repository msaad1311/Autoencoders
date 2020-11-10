# from cv2 import cv2
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def parameters(name,path):
    path = path.replace('\\','/')
    path_old = os.getcwd()
    os.chdir(path)
    vid = cv2.VideoCapture(name)
    fps=vid.get(cv2.CAP_PROP_FPS)
    width  = vid.get(cv2.CAP_PROP_FRAME_WIDTH) # float
    height = vid.get(cv2.CAP_PROP_FRAME_HEIGHT) # float
    
    os.chdir(path_old)
    
    return fps,int(width),int(height)
    
    
    
def video_creater(video_name,path,fps,width,height,ims):
    print('Working on Reconstructed Video')
    path = path.replace('\\','/')
    path_old = os.getcwd()

    os.chdir(path)

    images=[]
    names=[]
    count=0
    
    for i in tqdm(range(len(ims)),desc='Frames Saving'):
        cv2.imwrite('frame_{}.jpg'.format(i),ims[i])
    
    for i in tqdm(range(len(ims)),desc='Frame Reading'):
        im = cv2.imread('frame_{}.jpg'.format(i))
        name = 'frame_{}.jpg'.format(i)
        names.append(name)
        images.append(im)

    for i in tqdm(range(0,len(images)-1,2),desc='Concatenating Frames'):
        x1 = cv2.imread(names[i])
        x2 = cv2.imread(names[i+1])
        x3 = cv2.hconcat([x1,x2])
        try:
            os.remove(names[i])
            os.remove(names[i+1])
        except IndexError:
            print(i)
            break
        cv2.imwrite('frame_{}.jpg'.format(count),x3)
        count+=1

    video = cv2.VideoWriter(video_name, 0 , fps, (width,height))
    
    for n in names:
        video.write(cv2.imread(n))

    cv2.destroyAllWindows()
    video.release()
    
    for i in range(count):
        os.remove('frame_{}.jpg'.format(i))
    print('Reconstructed Frames removed.')

    os.chdir(path_old)
    return    

def original_video(video_name,saving_path,path,fps,data):
    print('Working on Original Video')
    path = path.replace('\\','/')
    saving_path = saving_path.replace('\\','/')
    
    path_old = os.getcwd()

    os.chdir(path)

    images=[]
    names=[]
    for i in tqdm(range(len(data)),desc='Frame Reading'):
        im = cv2.imread('frame_{}.jpg'.format(i))
        height, width = im.shape[:2]
        name = 'frame_{}.jpg'.format(i)
        names.append(name)
        images.append(im)
    
    video = cv2.VideoWriter(video_name, 0 , fps, (int(width),int(height)))
    
    for n in names:
        video.write(cv2.imread(n))

    cv2.destroyAllWindows()
    video.release()

    os.rename(os.path.join(os.getcwd(),video_name), 
    os.path.join(saving_path,video_name))
    
    os.chdir(path_old)
    return    