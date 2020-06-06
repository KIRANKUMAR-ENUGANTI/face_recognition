import cv2
import os
import datetime
camere_port = 0
image_dir="faceDataset/"
current_datetime=datetime.datetime.now()

def path_exists(imagepath):
    dir = os.path.dirname(imagepath)
    if not os.path.exists(dir):
        os.makedirs(dir)


face_id = input('enter your id')
vid_cam = cv2.VideoCapture(camere_port, cv2.CAP_DSHOW)

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
count = 0

path_exists(image_dir)

while (True):

    _, image_frame = vid_cam.read()

    gray = cv2.cvtColor(image_frame, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        count += 1
        cv2.rectangle(image_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        font = cv2.FONT_HERSHEY_COMPLEX_SMALL
        cv2.putText(image_frame,f"{face_id}-count{count}--{current_datetime}",(x-6,y-6),font, 0.8, (225,0,0), 1)


        cv2.imwrite(f"{image_dir}/User.{face_id}.{count}.jpg", gray[y:y + h, x:x + w])

        cv2.imshow('frame', image_frame)

    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

    elif count >= 100:
        print("Successfully Captured")
        break

vid_cam.release()

cv2.destroyAllWindows()
