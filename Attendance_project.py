import cv2
import numpy as np
import face_recognition
import os
import datetime

path = "images"
images = []
classNames = []

myList = os.listdir(path)
print(myList)

for cl in myList:
    curImg = cv2.imread(f"{path}/{cl}")
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)


#find encoding
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

encodeListKnown = findEncodings(images)
print("encoding complete")


#for attendnace

def makeAttendance(name):
    with open('attandance.csv','r+') as f:
        mydatalist = f.readline()
        namelist= []
        for line in mydatalist:
            entry = line.split(',')
            namelist.append(entr[0])
        if name not in namelist:
            now = datatime.now()
            dtstring = now.strftime('%H:%M:%S')
            f.writelines(f"\n{name},{dtstring}")


#makeAttendance("a")


cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgS = cv2.resize(img,(0,0),None,0.25,0.25)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #converting it to grb

    facesCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS,facesCurFrame)

    for encodeFace,faceLoc in zip(encodeCurFrame,facesCurFrame):#it will thake them in sequance
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown,encodeFace) #Lowest distance would be good match

        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4 #becuase we scalled the image before so we multipy with 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            makeAttendance(name)
    cv2.imshow("Web Cam", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#when everything is done release the screen
cap.release()
cv2.destroyAllWindows()




