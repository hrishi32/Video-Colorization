import cv2
import sys

def frameExtractor(path):
    videoObject = cv2.VideoCapture(path)

    count = 1
    success = 1
    while success:
        success, image = videoObject.read()
        cv2.imwrite(sys.argv[2]+"/frame%d.jpg" % count, image[:,:480,:]) 
  
        count += 1

    return count

framesExtracted = frameExtractor(sys.argv[1])
