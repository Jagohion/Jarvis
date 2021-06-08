import cv2 as cv
import time
import mediapipe as mp
import os
import math
import numpy as np
#pip install pycaw encapsulates the following lib
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

class handDetector():
    def __init__(self, mode=False,maxHands=2,detectionCon=0.5,trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode,self.maxHands,self.detectionCon,self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        self.volume = cast(interface,POINTER(IAudioEndpointVolume))
        self.volRange = self.volume.GetVolumeRange()

        self.path = os.path.abspath(__file__)
        self.path = os.path.dirname(self.path)

    def findHands(self,img,draw=True):
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img,handLms,self.mpHands.HAND_CONNECTIONS)
        return img
    
    def findPosition(self,img,handNo=0,draw=True):
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h,w,c=img.shape
                cx,cy=int(lm.x*w),int(lm.y*h)
                lmList.append([id,cx,cy])
                if draw:
                    cv.circle(img,(cx,cy),15,(255,0,255),cv.FILLED)
        
        return lmList 

    def setVolume(self,img,lmList,draw=True):
        if len(lmList) != 0:
            x1,y1 = lmList[4][1],lmList[4][2]   #thumb tip
            x2,y2 = lmList[12][1],lmList[12][2]   #middle finger tip
            cx,cy = (x1+x2)/2,(y1+y2)/2
            length = math.hypot(x2-x1,y2-y1)
            cv.line(img,(x1,y1),(x2,y2),(255,0,255),3)
            #handrange 50 - 150 change according to hand size
            vol = np.interp(length,[50,150],[self.volRange[0],self.volRange[1]])
            self.volume.SetMasterVolumeLevel(vol,None)

    #take a screenshot and store it in the given path
    def screenShot(self,lmList,img,path='myscreenshot.png'):
        point1 = lmList[8]
        point2 = lmList[2]
        point3 = lmList[4]
        try:
            m1 = (point1[2]-point2[2])/(point1[1]-point2[1])
            m2 = (point3[2]-point2[2])/(point3[1]-point2[1])
            cv.line(img,(point1[1],point1[2]),(point2[1],point2[2]),(255,0,255),2)
            cv.line(img,(point3[1],point3[2]),(point2[1],point2[2]),(255,0,255),2)
        except ZeroDivisionError:
            cv.imwrite(path,img)
            return True
        angle = math.atan((m2-m1)/(1+m2*m1))
        angle = abs(angle)*180/math.pi
        if(angle>84):
            cv.imwrite(path,img)
            return True
def main():
    pTime = 0
    cTime = 0
    cap = cv.VideoCapture(0)
    detector = handDetector()

    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList = detector.findPosition(img,draw=False) 
        #detector.setVolume(img,lmList)
        if len(lmList)!=0:
            detector.screenShot(lmList,img)
            #fps
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        msg = "FPS:"+str(int(fps))
        cv.putText(img,msg,(10,70),cv.FONT_HERSHEY_SIMPLEX,1,(255,255,0),3)

        cv.imshow("Image",img)
        if cv.waitKey(20) & 0xFF==ord('d'):
            break
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
