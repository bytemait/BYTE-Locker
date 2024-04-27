from flask import Flask, request, render_template, jsonify
import cv2
import numpy as np

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))
from detector import FaceRecognizer

from barcode import BarcodeDecoder

app = Flask(__name__)

# Functionality for the /Face page
@app.route('/Face', methods=['GET', 'POST'])
def face_detection():
    recog = FaceRecognizer()

    if request.method == 'POST':
        file = request.files['image']

        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        res = recog.detect(img)
        print(res)

        return f"{res} detected"
    
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
    app.run(debug=True)