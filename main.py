import matplotlib.pyplot as plt
import cv2

import os
 
   
data_dir = "C:/Users/Vasanth M/jupyter_codes/face_det_oc/data"
model_dir = "C:/Users/Vasanth M/jupyter_codes/face_det_oc/model"

face_detector = cv2.CascadeClassifier(os.path.join(model_dir, "haarcascade_frontalface_default.xml"))


for img_path in os.listdir(data_dir):

    img = cv2.imread(os.path.join(data_dir, img_path))

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    faces = face_detector.detectMultiScale(img_gray, minNeighbors = 20)

    plt.figure()

    for face in faces :
        x1, y1, w, h = face

        img = cv2.rectangle(img, (x1, y1), (x1+w, y1+h),(0,255,0), 10)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)

plt.show()