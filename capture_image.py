import cv2
import os
import datetime,time


class Capture:
    if __name__ == "__main__":
        def __init__(self):
            self.camera_port = 0
            self.image_dir = "faceDataset/"
            self.current_datetime = datetime.datetime.now()
            self.face_id = input('enter your id')
            self.user_name = input("enter you name")
            self.vid_cam = cv2.VideoCapture(self.camera_port, cv2.CAP_DSHOW)
            self.face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
            self.count = 1
            self.capture_image()

    def path_exists(self, main_dir, sub_dir=""):
        dir = os.path.dirname(main_dir)
        if not os.path.exists(dir):
            os.makedirs(dir)
            return
        sub_dir = os.path.join(main_dir, sub_dir)
        if not os.path.exists(sub_dir):
            os.mkdir(sub_dir)
            return

    def capture_image(self):
        self.path_exists(self.image_dir)
        self.path_exists(self.image_dir, self.face_id)
        user_folder = os.path.join(self.image_dir, self.face_id)
        while (True):
            _, image_frame = self.vid_cam.read()

            gray = cv2.cvtColor(image_frame, cv2.COLOR_BGR2GRAY)

            faces = self.face_detector.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:

                cv2.rectangle(image_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                font = cv2.FONT_HERSHEY_COMPLEX_SMALL
                cv2.putText(image_frame, f"{self.user_name}/{self.face_id}-count{self.count}--{self.current_datetime}",
                            (x - 6, y - 6),
                            font, 0.8, (225, 0, 0), 1)
                self.path_exists(self.image_dir, self.face_id)

                if cv2.waitKey(1) & 0xFF == ord('c'):
                    cv2.imwrite(f"{user_folder}/{self.user_name}.{self.face_id}.{self.count}.jpg", image_frame[y:y + h, x:x + w])
                    cv2.rectangle(image_frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
                    self.count += 1

                cv2.imshow('frame', image_frame)
            if self.count >= 6:
                print("Successfully Captured")
                break

        self.vid_cam.release()

        cv2.destroyAllWindows()


c = Capture()
