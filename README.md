# Face Recognition with OpenCV

## Installation
Install the required packages using pip:

```bash
pip install -r requirements.txt
```

If you face cmake error you can download dlib of your python version from [here](https://github.com/z-mahmud22/Dlib_Windows_Python3.x).

```bash
pip install downloaded_file_name.whl
pip install -r requirements.txt
```

## Usage
1. Create a folder for each person in the `/training` directory and add their pictures (PNG, JPG, or JPEG formats).
2. Run the following commands to encode the known faces:

```python
from face_recognizer import FaceRecognizer

recognizer = FaceRecognizer()
recognizer.encode_known_faces()
```

3. To test the recognition, add test images to the `/testing` directory and run:

```python
recognizer.test("testing")
```

4. For live feed face recognition, run:

```python
recognizer.live_feed()
```

```