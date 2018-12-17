# Computer Vision

## Dependencies

- Python 3
- [Davis King's Dlib with CUDA support (optional)](https://github.com/davisking/dlib)
- [Adam Geitgey's face_recognition](https://github.com/ageitgey/face_recognition)
- Scikit-Learn
- OpenCV

## Project Structure

Project structure:

```
final-project-computer-vision/
├── foto/
│   │   ├── person1
│   │   │   ├── something.jpeg
│   │   │   ├── something.png
│   │   ├── person2
│   │   │   ├── something.jpeg
│   │   │   ├── something.png
│   │   ├── ...
├── foto_asli/
│   │   ├── person1
│   │   │   ├── something.jpeg
│   │   │   ├── something.png
│   │   ├── person2
│   │   │   ├── something.jpeg
│   │   │   ├── something.png
│   │   ├── ...
└── main.py
└── readme.md
└── trained_model.clf
└── trainmodel.py
```

**foto/** : LFW funneled dataset combined with personal dataset.
**foto_asli/** : Personal dataset.
**main.py** : Main program. Run this file after training your model.
**readme.md** : This file. Contains any information you need.
**trained_model.clf** : Pre-trained model to use.
**trainmodel.py** : Run this file if you want to train the model on your own dataset.

## Usage

- Install the dependencies
- Add your dataset to folder **foto** (optional)
- Run **trainmodel.py** to generate model (optional, required if you want to re-train the model with your own dataset)
- Run **main.py** file

## dlib with CUDA support

I'm using dlib with CUDA support to make the program run faster on my personal computer. If you don't have GPU with CUDA support or installing dlib without CUDA support, then modify the **main.py** file :

##### Original main.py (line 15)

```py
X_face_locations = face_recognition.face_locations(X_img_path, model="cnn")
```

##### Modified main.py (line 15)

```python
X_face_locations = face_recognition.face_locations(X_img_path)
```

It will change from **CNN** to **HOG**. You might want to change the **trainmodel.py** too, then re-train the model if the expected result is not good enough for you.