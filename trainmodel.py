import math
from sklearn import neighbors
import os
import os.path
import pickle
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder
import time
from collections import Counter

def train(train_dir, model_save_path=None, n_neighbors=None, knn_algo='ball_tree', verbose=False):
    """
    Trains a k-nearest neighbors classifier for face recognition.
    """
    X = []
    y = []
    known_faces = 0
    ignored_images = 0
    total_images = 0

    start_time = time.time()
    for class_dir in os.listdir(train_dir):
        if not os.path.isdir(os.path.join(train_dir, class_dir)):
            continue

        for img_path in image_files_in_folder(os.path.join(train_dir, class_dir)):
            image = face_recognition.load_image_file(img_path)
            total_images += 1
            face_location = face_recognition.face_locations(image, model="cnn")

            if len(face_location) != 1:
                ignored_images += 1
                if verbose:
                    print("Image {} not suitable for training: {}".format(img_path, ("Didn't find a face" if len(face_location) < 1 else "Found more than one face")))
            else:
                if verbose:
                    print("Found a face on {}".format(img_path))
                known_faces += 1
                X.append(face_recognition.face_encodings(image, face_location)[0])
                y.append(class_dir)

    if n_neighbors is None:
        n_neighbors = int(round(math.sqrt(len(X))))
        if verbose:
            print("Choosing n_neighbors automatically: ", n_neighbors)

    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
    knn_clf.fit(X, y)
    elapsed_time = time.time() - start_time
    print("It takes {:.3f} to completely trains the model.".format(elapsed_time))

    if model_save_path is not None:
        with open(model_save_path, 'wb') as f:
            pickle.dump(knn_clf, f)

    identified_person = len(Counter(y).keys())
    print("Saved as {} => {:,} total datasets, {:,} identified faces, {:,} ignored images, and {:,} identified person."
          .format(model_save_path, total_images, known_faces, ignored_images, identified_person))

    return knn_clf


if __name__ == '__main__':
    print("Training classifier... Please wait...")
    classifier = train(train_dir="foto", model_save_path="trained_model.clf", n_neighbors=2, verbose=True)
    print("Training complete!")