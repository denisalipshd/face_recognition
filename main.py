import os
import face_recognition
from sklearn import neighbors
import pickle
import cv2

def train():
    x = []
    y = []
    
    for name in os.listdir('train_dir'):
        person_folder = os.path.join('train_dir', name)

        if not os.path.isdir(person_folder):
            continue

        for filename in os.listdir(person_folder):
            file_path = os.path.join(person_folder, filename)

            image = face_recognition.load_image_file(file_path)
            image_location = face_recognition.face_locations(image)
            
            if len(image_location) != 1:
                print("Gambar tidak ada muka yang terdeteksi:", file_path)
                continue
            
            x.append(face_recognition.face_encodings(image, known_face_locations=image_location)[0])
            y.append(name)
            
    model = neighbors.KNeighborsClassifier(n_neighbors=2)
    model.fit(x, y)
    
    with open('face_recognition_model.clf', 'wb') as f:
        pickle.dump(model, f)

if __name__ == '__main__':
    # Jalankan sekali untuk training, jika model belum ada
    # train()                 

    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        print("Tidak dapat membuka kamera")
        exit()
    
    with open('face_recognition_model.clf', 'rb') as f:
        model = pickle.load(f)
    
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        face_location = face_recognition.face_locations(small_frame)

        if len(face_location) == 0:
            continue
           
        face_encoding = face_recognition.face_encodings(small_frame, known_face_locations=face_location)
        face_names = model.predict(face_encoding)
                
        for (top, right, bottom, left), name in zip(face_location, face_names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)
        
        cv2.imshow('Video', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()
