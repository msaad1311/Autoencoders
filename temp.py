from cv2 import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

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
    
    
    
def video_creater(video_name,path,fps,width,height):
    path = path.replace('\\','/')
    path_old = os.getcwd()

    os.chdir(path)

    images=[]
    names=[]

    for i in range(739):
        image = cv2.imread('frame_{}.jpg'.format(i))
        name = 'frame_{}.jpg'.format(i)
        names.append(name)
        images.append(image)

    video = cv2.VideoWriter(video_name, 0 , fps, (width,height))

    for image in names:
        video.write(cv2.imread(image))

    cv2.destroyAllWindows()
    video.release()
    
    os.chdir(path_old)
    return

if __name__ == "__main__":
    
    path_original = os.getcwd()
    path_frames = r'C:\Users\Saad.LAKES\Desktop\Autoencoders\ReconVideo_Frames'
    name_original = 'bunny_video.mp4'
    name_reconstructed = 'reconstructed.avi'
    
    fps,width,height = parameters(name_original,path_original)
    print(fps,width,height)
    video_creater(name_reconstructed,path_frames,fps,128,128)
