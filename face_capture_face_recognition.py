import cv2
import face_recognition
from PIL import Image
import os
import datetime
import numpy as np
import pickle
import xlwriter

#reading the faces_encoded from pkl
file="faces_encoded_pickle.pkl"
f_handle=open(file,'rb')
known_face_encodings=pickle.load(f_handle)
f_handle.close()

file="faces_names_pickle.pkl"
f_handle=open(file,'rb')
known_face_names=pickle.load(f_handle)
video_capture = cv2.VideoCapture(0)
f_handle.close()

face_locations = []
face_encodings = []
face_names = []
process_this_frame = True


def actual_index(index):
    if index % 5 == 0:
        region = index // 5
    else:
        region = index % 15
        region = 15 - region
        region = index + region
        region = region // 15
    return region

def draw_border(img, pt1, pt2, color, thickness, r, d):
    x1,y1 = pt1
    x2,y2 = pt2

    # Top left
    cv2.line(img, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)

    # Top right
    cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)

    # Bottom left
    cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
    cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)

    # Bottom right
    cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
    cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)
while True:
    ret, frame = video_capture.read()

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    rgb_small_frame = small_frame[:, :, ::-1]

    if process_this_frame:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # If a match was found in known_face_encodings, just use the first one.
            if True in matches:
                index = matches.index(True)
                index = actual_index(index)
                name = known_face_names[index]

            face_names.append(name)

    process_this_frame = not process_this_frame

    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        if name != "Unknown":
            draw_border(frame, (left, top), (right, bottom), (0, 255, 0), 1,10,10)
            font = cv2.FONT_HERSHEY_COMPLEX_SMALL
            cv2.putText(frame, name, (left,top-6), font, 0.8, (255, 255, 255), 1)
            # xlwriter.output('attendance', 'class1', 4, name, 'yes')
        else:
            draw_border(frame, (left, top), (right, bottom), (0,0,255), 1,10,10)
            font = cv2.FONT_HERSHEY_COMPLEX_SMALL
            cv2.putText(frame, name, (left, top - 6), font, 0.8, (255, 255, 255), 1)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
