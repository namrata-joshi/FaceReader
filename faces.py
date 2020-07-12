import  numpy as np
import cv2
import  pickle
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

labels = {"person_name":1}
with open("label.pickle", "rb") as f:
    og_labels = pickle.load(f)
    labels = {v:k for k,v in og_labels.items()}

cap = cv2.VideoCapture(0)

while(True):
    # capture frame-by-read
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.1,5)

    for(x,y,w,h) in faces:
        #print(x,y,w,h)
        #region of interest ROI
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h,x:x+w]

        #recognize?... deep learned model keras, scikit learn, tensorflow,
        id_, conf = recognizer.predict(roi_gray)
        if conf>=50:# and conf<=85:
                print(id_)
                print(labels[id_])
                font = cv2.FONT_HERSHEY_COMPLEX
                name = labels[id_]
                color =(255,255,255)
                stroke =2
                cv2.putText(frame,name,(x,y), font, 1, color, stroke, cv2.LINE_AA)
        img_item = 'my-img.png'
        cv2.imwrite(img_item, roi_gray)
        #draw a rectangle over face
        color = (255,0,0) #BGR
        #stroke of the rectangle
        stroke = 2
        end_chord_x = x+w
        end_chord_y = y+h
        cv2.rectangle(frame,(x,y),(end_chord_x,end_chord_y), color, stroke)

    # display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(20) & 0xFF==ord('q'):
        break;


return_value, image = cap.read()
print("We take a picture of you, check the folder")
cv2.imwrite("my-img.png", image)
cap.release()
cv2.destroyAllWindows()
