import cv2
import os
import face_recognition
import numpy as np
from capture_image import Capture
import pickle

faces_encoded = []
faces_names = []
image_names=[]
main_image_dir='faceDataset/'


for (_,_, file) in os.walk('faceDataset', topdown=False):
    image_names.extend(file)


for image_name in image_names:
    user_name,id,_=image_name.split('.',2)
    image_path=f"{main_image_dir}{id}/{image_name}"
    user_image = face_recognition.load_image_file(image_path)
    user_face_encoding = face_recognition.face_encodings(user_image)[0]
    faces_encoded.append(user_face_encoding)
    if user_name not in faces_names:
        faces_names.append(user_name)

file="faces_encoded_pickle.pkl"
f_handle=open(file,'wb')
pickle.dump(faces_encoded,f_handle)
f_handle.close()
file='faces_names_pickle.pkl'
f_handle=open(file,'wb')
pickle.dump(faces_names,f_handle)