
import os
from PIL import Image
import numpy as np
import cv2
import pickle
#import cv2.face.facerec.hpp
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')

BASE_dir = os.path.dirname(os.path.abspath(__file__))

print(BASE_dir)

recognizer = cv2.face.LBPHFaceRecognizer_create()

image_dir = os.path.join(BASE_dir,'Image')

X_train = []
Y_labels = []
current_id = 0

label_ids = {}
for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith('png')or file.endswith("jpg") :
            path = os.path.join(root, file)
            label = os.path.basename(os.path.dirname(path)).replace(" ","-").lower()
            
        if not label in label_ids:
           label_ids[label]=current_id
           current_id+=1
        id_ = label_ids[label]
        #print(label_ids)
        pil_image = Image.open(path).convert("L") # grayscale
        size =(550,550)
        final_size = pil_image.resize(size, Image.ANTIALIAS)
        image_array = np.array(pil_image,'uint8')
        #print(image_array)
        faces = face_cascade.detectMultiScale(image_array,1.5,5)

        for(x,y,w,h) in faces:
            roi = image_array[y:y+h, x:x+w]
            X_train.append(roi)
            Y_labels.append(id_)


#print(Y_labels)
#print(X_train)


with open("label.pickle", "wb") as f:
    pickle.dump(label_ids,f)

print(type(Y_labels))


recognizer.train(X_train, np.array(Y_labels))
recognizer.save("trainer.yml")


