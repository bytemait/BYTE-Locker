from flask import Flask, request, render_template, jsonify
import cv2
import numpy as np
import time
import os
import sys
from base64 import b64encode

sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))
from scripts.detector import FaceRecognizer
from scripts.barcode import BarcodeDecoder
import sqlite3

app = Flask(__name__)

# Functionality for the /Face page

def addToDB(id: str, camera_id: str, timestamp: str, image):
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    image = b64encode(image).decode("utf-8")
    c.execute('INSERT INTO detections (person_id, camera_id, timestamp, image) VALUES (?, ?, ?, ?)', (id, camera_id, timestamp, image))
    conn.commit()
    conn.close()

def setupDB():
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute('CREATE TABLE IF NOT EXISTS detections (person_id TEXT, camera_id TEXT, timestamp TEXT, image BLOB)')
    conn.commit()
    conn.close()

@app.route('/Face', methods=['GET', 'POST'])
def face_detection():
    recog = FaceRecognizer()

    if request.method == 'POST':
        file = request.files['image']

        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        person_id = recog.detect(img)[0]
        print(person_id)
        addToDB(person_id, '1', time.strftime('%Y-%m-%d %H:%M:%S'), img) # ToDo: replace '1' with the camera ID 
        return f"{person_id} detected"
    
    else:
        return render_template('face.html')

@app.route('/Barcode', methods=['GET', 'POST'])
def barcode_detection():
    dec = BarcodeDecoder()

    if request.method == 'POST':
        file = request.files['image']

        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        res = dec.decode(img)
        print(res)

        return f"{res} decoded"
    
    else:
        return render_template('barcode.html')
    
if __name__ == '__main__':
    setupDB()
    app.run(debug=True)