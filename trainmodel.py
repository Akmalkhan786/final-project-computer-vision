import math
from sklearn import neighbors
import os
import os.path
import pickle
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder

def train(train_dir, model_save_path=None, n_neighbors=None, knn_algo='ball_tree', verbose=False):
    """
    Trains a k-nearest neighbors classifier for face recognition.
    """
    X = []
    y = []

    for class_dir in os.listdir(train_dir):
        if not os.path.isdir(os.path.join(train_dir, class_dir)):
            continue

        for img_path in image_files_in_folder(os.path.join(train_dir, class_dir)):
            image = face_recognition.load_image_file(img_path)
            face_location = face_recognition.face_locations(image, number_of_times_to_upsample=0, model="cnn")

            if len(face_location) != 1:
                print("Image {} not suitable for training: {}".format(img_path, ("Didn't find a face" if len(face_location) < 1 else "Found more than one face")))
            else:
                print("Found a face on {}".format(img_path))
                X.append(face_recognition.face_encodings(image, face_location)[0])
                y.append(class_dir)

    if n_neighbors is None:
        n_neighbors = int(round(math.sqrt(len(X))))
        if verbose:
            print("Choosing n_neighbors automatically: ", n_neighbors)

    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
    knn_clf.fit(X, y)

    if model_save_path is not None:
        with open(model_save_path, 'wb') as f:
            pickle.dump(knn_clf, f)

    return knn_clf


if __name__ == '__main__':
    print("Training CNN + KNN classifier...")
    classifier = train(train_dir="foto", model_save_path="trained_model.clf", n_neighbors=2, knn_algo='kd_tree')
    print("Training complete!")