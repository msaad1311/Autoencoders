# from cv2 import cv2
import cv2
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
    
    
    
def video_creater(video_name,path,fps,width,height,types):
    path = path.replace('\\','/')
    path_old = os.getcwd()

    os.chdir(path)

    images=[]
    names=[]
    
    for i in range(739):
        image = cv2.imread('frame_{}.jpg'.format(i))
        if image.shape != (width,height,3):
            image = cv2.resize(image,(width,height))
            cv2.imwrite('frame_{}.jpg'.format(i+740),image)
        name = 'frame_{}.jpg'.format(i)
        names.append(name)
        images.append(image)
    
    video = cv2.VideoWriter(video_name, 0 , fps, (width,height))
    
    for image in names:
        video.write(cv2.imread(image))

    cv2.destroyAllWindows()
    video.release()
    
    if types=='Original':
        for n in names:
            os.remove(n)

    os.chdir(path_old)
    return      

if __name__ == "__main__":
    
    path = os.getcwd()
    path_reconstructed = r'C:\Users\Saad.LAKES\Desktop\Autoencoders\ReconVideo_Frames'
    path_original = r'C:\Users\Saad.LAKES\Desktop\Autoencoders\Video_Frames'
    # path_original = '/Users/saad/Desktop/Projects/Autoencoders/Temp' #for Mac

    name = 'bunny_video.mp4'
    name_reconstructed = 'reconstructed.avi'
    name_original = 'original.avi'
    
    fps,width,height = parameters(name,path)
    print(fps,width,height)
    width,height = 128,128
    video_creater(name_reconstructed,path_reconstructed,fps,width,height,'Reconstructed') # Reconstructed video
    video_creater(name_original,path_original,fps,width,height,'Original') # Original video
