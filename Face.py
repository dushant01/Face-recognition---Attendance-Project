import cv2
import numpy as np
import face_recognition

imgRan = face_recognition.load_image_file('image_basic/ranbir2.jpg')
imgRan = cv2.cvtColor(imgRan,cv2.COLOR_BGR2RGB)
imgTest = face_recognition.load_image_file('image_basic/randir test.jpg')
imgTest = cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)

#findingt the location and ploting rectangle around image
faceLoc = face_recognition.face_locations(imgRan)[0]
encodeRan = face_recognition.face_encodings(imgRan)[0]
cv2.rectangle(imgRan,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)

#finding the location and ploting rectangle around test image
faceLocTest = face_recognition.face_locations(imgTest)[0]
encodeRanTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,255),2)

#comparning the images
results= face_recognition.compare_faces([encodeRan],encodeRanTest)
faceDis = face_recognition.face_distance([encodeRan],encodeRanTest) #llower the distance more the match
print(results,faceDis)
cv2.putText(imgTest,f'{results} {round(faceDis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)

cv2.imshow("ran musk",imgRan)
cv2.imshow("ran Test",imgTest)

cv2.waitKey(0)