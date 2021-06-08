import cv2 as cv
import numpy as np
import face_recognition
import time
import os
import csv
from datetime import datetime

class FaceRecognition():
    def __init__(self):
        self.path = os.path.abspath(__file__)
        self.path = os.path.dirname(self.path)
        self.imgpath = os.path.join(self.path,'Faces')
        self.markpath = os.path.join(self.path,'MarkTime.csv')
        self.namepath = os.path.join(self.path,'NameList.txt')
        self.encodepath = os.path.join(self.path,'data.npy')
        
        self.images = []
        self.classNames = []
        self.encodeListKnown = []
        
    
    @property
    def initialize(self):
        if not os.path.exists(self.imgpath):
            os.mkdir(self.imgpath)
            with open(self.markpath, "w") as f:
                writer = csv.writer(f)
            myfile = open(self.namepath,'w+')
            myfile.close()

    #find the encodings of a given img
    def findEncodings(self,images):
        encodeList = []
        for img in self.images:
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            encode = face_recognition.face_encodings(img)[0]
            encodeList.append(encode)
        return encodeList

    #scanning a new face to find and store its encodings
    def initScan(self):
        myList = os.listdir(self.imgpath)
        for cl in myList:
            curImg = cv.imread(f'{self.imgpath}/{cl}')
            self.images.append(curImg)
            self.classNames.append(os.path.splitext(cl)[0])
        self.encodeListKnown = self.findEncodings(self.images)

        np.save(self.encodepath,self.encodeListKnown)
        np.savetxt(self.namepath,self.classNames,fmt="%s")

    def postScan(self):
        
        self.classNames = np.loadtxt(self.namepath,dtype = str)
        self.encodeListKnown = np.load(self.encodepath)
        
    #adding a new face (note it will give an error if no face is detected)
    def addImg(self,videocap = 0):
        cap = cv.VideoCapture(videocap)
        name = input("Enter your name: ")
        print('Collecting images of {}'.format(name))
        print('press d to take a screenshot')
        while True:
            success, img = cap.read()
            cv.imshow('screenshot',img)
            if cv.waitKey(20) & 0xFF==ord('d'):
                imgname = os.path.join(self.imgpath,name+'.jpg')
                cv.imwrite(imgname, img)
                break
        cv.destroyAllWindows()
        
    #give the faceframe a fancier remake
    def fancyDraw(self,img, faceLoc,ratio,name,length=30,thickness=5):
        y,x1,y1,x = faceLoc
        y,x1,y1,x = int(y*ratio),int(x1*ratio),int(y1*ratio),int(x*ratio)
        
        #cv.rectangle(img,(x,y1-35),(x1,y1),(0,255,0),cv.FILLED)
        cv.putText(img,name,(x+6,y-10),cv.FONT_HERSHEY_PLAIN,2,(255,255,255),2)
        cv.rectangle(img,(x,y),(x1,y1),(255,0,255),1)
        #top left
        cv.line(img,(x,y),(x+length,y),(255,0,255),thickness)
        cv.line(img,(x,y),(x,y+length),(255,0,255),thickness)

        #top right   
        cv.line(img,(x1,y),(x1-length,y),(255,0,255),thickness)
        cv.line(img,(x1,y),(x1,y+length),(255,0,255),thickness)

        #bottom right
        cv.line(img,(x1,y1),(x1-length,y1),(255,0,255),thickness)
        cv.line(img,(x1,y1),(x1,y1-length),(255,0,255),thickness)

        #bottom left
        cv.line(img,(x,y1),(x+length,y1),(255,0,255),thickness)
        cv.line(img,(x,y1),(x,y1-length),(255,0,255),thickness)

        return img
    
    def markAttendance(self,name):
        with open(self.markpath,'r+') as f:
            myDataList = f.readlines()
            nameList = []
            for line in myDataList:
                entry = line.split(',')
                nameList.append(entry[0])
            if name not in nameList:
                now = datetime.now()
                dtString = now.strftime('%H:%M:%S')
                f.writelines(f'\n{name},{dtString}')

                
        
    def findFace(self,img,imgS,offset,mark=True):
        faceCurFrame = face_recognition.face_locations(imgS)
        encodeCurFrame = face_recognition.face_encodings(imgS,faceCurFrame)

        for encodeFace,faceLoc in zip(encodeCurFrame,faceCurFrame):
            matches = face_recognition.compare_faces(self.encodeListKnown,encodeFace)
            faceDis = face_recognition.face_distance(self.encodeListKnown,encodeFace)
            
            matchIndex = np.argmin(faceDis)

            if matches[matchIndex]:
                name = self.classNames[matchIndex]
                #print(img)
                img = self.fancyDraw(img,faceLoc,offset,name)
                self.markAttendance(name,self.markpath)
        cv.imshow("Webcam",img)

def main():
    
    detector = FaceRecognition()
    detector.initialize
    
    #detector.addImg()
    #detector.initScan()
    detector.postScan()
    
    cap = cv.VideoCapture(0)
    
    while True:
        offset = 0.25
        success, img = cap.read()
        imgS = cv.resize(img,(0,0),None,offset,offset)
        imgS = cv.cvtColor(imgS, cv.COLOR_BGR2RGB)

        detector.findFace(img,imgS,1/offset)

        if cv.waitKey(20) & 0xFF==ord('d'):
            break
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
    
