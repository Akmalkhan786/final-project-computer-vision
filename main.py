import face_recognition, cv2, pickle
# import knn

def predict(X_img_path, knn_clf=None, model_path=None, distance_threshold=0.6):
    """
    Recognizes faces in given image using a trained KNN classifier
    """
    if knn_clf is None and model_path is None:
        raise Exception("Harus ada model yang dipakai, bisa menggunakan argumen 'knn_clf' ataupun 'model_path'")

    if knn_clf is None:
        with open(model_path, 'rb') as f:
            knn_clf = pickle.load(f)

    X_face_locations = face_recognition.face_locations(X_img_path)

    if len(X_face_locations) == 0:
        return []

    faces_encodings = face_recognition.face_encodings(X_img_path, known_face_locations=X_face_locations)

    # Use KNN model
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]

    return [(pred, loc) if rec else ("Unidentified", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]


if __name__ == '__main__':
    video_capture = cv2.VideoCapture(0)
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    face_locations = []
    face_encodings = []
    face_names = []
    # process_this_frame = True

    while True:
        ret, frame = video_capture.read()
        frame = cv2.flip(frame, 1)
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]

        # if process_this_frame:
        predictions = predict(rgb_small_frame, model_path="trained_model.clf")
        # face_names = []
        for name, (top, right, bottom, left) in predictions:
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            cv2.rectangle(frame, (left, top), (right, bottom+10), (0, 255, 0), 2)
            cv2.rectangle(frame, (left, bottom + 35), (right, bottom), (0, 255, 0), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left, bottom + 20), font, 0.8, (0, 0, 255), 1)
            print("- Terdeteksi {} di ({}, {})".format(name, left, top))

        # Display the resulting image
        cv2.imshow('Realtime Face Identifier', frame)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()