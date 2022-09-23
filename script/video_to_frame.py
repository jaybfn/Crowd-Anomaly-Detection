# importing libraries
import cv2
import os

def create_dir(path):
        isExist = os.path.exists(path)
        if not isExist:
            os.makedirs(path, exist_ok = False)
            print("New directory is created")

def getFrame(sec, file_count):
    vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
    hasFrames,image = vidcap.read()
    if hasFrames:
        cv2.imwrite(image_path+str(file_count)+"image"+str(count)+".jpg", image)     # save frame as JPG file
    return hasFrames


if __name__ == '__main__':

    # folder name
    folder = 'frames_val'

    # creating directory to save all the frame
    frame_dir = '../data'+'/'+folder
    create_dir(frame_dir)

    # path where the video data is stores
    video_source_path = '../data/validation_videos'

     # listing number of videos in the folder
    videos=os.listdir(video_source_path)
    print("Found ",len(videos)," validation_videos")

    # path to store images
    image_path = frame_dir+'/'

    for file_count,video in enumerate(videos):
        vidcap = cv2.VideoCapture('../data/validation_videos/'+ video)
        sec = 0
        frameRate = 2 #//it will capture image in each 0.5 second
        count=1
        success = getFrame(sec, file_count)
        while success:
            count = count + 1
            sec = sec + frameRate
            sec = round(sec, 2)
            success = getFrame(sec, file_count)


