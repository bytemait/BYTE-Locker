import face_recognition
import cv2
import pickle

from pathlib import Path
from collections import Counter

import os
import time

class FaceRecognizer:
    def __init__(self, encodings_location=Path("scripts/output/encodings.pkl")):
        self.encodings_location = encodings_location

    def encode_known_faces(self, model="hog"):
        names = []
        encodings = []
        for filepath in Path("training").glob("*/*"):
            name = filepath.parent.name
            image = face_recognition.load_image_file(filepath)

            face_locations = face_recognition.face_locations(image, model=model)
            face_encodings = face_recognition.face_encodings(image, face_locations)

            for encoding in face_encodings:
                names.append(name)
                encodings.append(encoding)

        name_encodings = {"names": names, "encodings": encodings}
        with self.encodings_location.open(mode="wb") as f:
            pickle.dump(name_encodings, f)

    def recognize_faces(self, image, model="hog"):
        names = []
        start = time.time()
        with self.encodings_location.open(mode="rb") as f:
            loaded_encodings = pickle.load(f)

        input_face_locations = face_recognition.face_locations(image, model=model)
        input_face_encodings = face_recognition.face_encodings(image, input_face_locations)

        img = image

        for bounding_box, unknown_encoding in zip(input_face_locations, input_face_encodings):
            name = self._recognize_face(unknown_encoding, loaded_encodings)
            if not name:
                name = "Unknown"
            names.append(name)
            print(name, bounding_box)
            img = self._add_data(img, name, bounding_box)

        return (img, names)

    def _add_data(self, input_image, name, bounding_box):
        top, right, bottom, left = bounding_box

        img = cv2.rectangle(input_image, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(img, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        return img

    def show(self, img):
        max_height = 800
        height, width = img.shape[:2]

        if height > max_height:
            scale_factor = max_height / height
            img = cv2.resize(img, (int(width * scale_factor), max_height))

        cv2.imshow("Result", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def _recognize_face(self, unknown_encoding, loaded_encodings):
        boolean_matches = face_recognition.compare_faces(loaded_encodings["encodings"], unknown_encoding)
        votes = Counter(name for match, name in zip(boolean_matches, loaded_encodings["names"]) if match)
        if votes:
            return votes.most_common(1)[0][0]

    def test(self, folder_path):
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.png', '.jpg', 'jpeg')):
                self.show(self.recognize_faces(cv2.imread(os.path.join(folder_path, filename)))[0])

    def live_feed(self):
        cap = cv2.VideoCapture(0)
        start_time = time.time()
        frame_count = 0

        while True:
            start = time.time()
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            current_time = time.time()
            elapsed_time = current_time - start_time
            if elapsed_time >= 1:
                fps = frame_count / elapsed_time
                frame_count = 0
                start_time = time.time()
            else:
                fps = frame_count / (elapsed_time + 0.0001)

            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow('Video', self.recognize_faces(frame)[0])

            print(f"{(time.time() - start):.2f} s")
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def detect(self,img):
        res = self.recognize_faces(img)
        # self.show(res[0])

        return res[1]
    
if __name__ == '__main__':
    recognizer = FaceRecognizer()
    recognizer.encode_known_faces()
#     # recognizer.test("testing")
#     # recognizer.live_feed()
    # recognizer.detect(cv2.imread("testing/unknown.jpeg"))