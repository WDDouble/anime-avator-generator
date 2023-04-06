import cv2
import os
from tqdm import tqdm

def detect_faces(image, classifier):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces

def save_faces(image, faces, output_dir, image_name):
    for i, (x, y, w, h) in enumerate(faces):
        face_img = image[y:y+h, x:x+w]
        cv2.imwrite(os.path.join(output_dir, f"{image_name}_face.jpg"), face_img)

input_dir = "data/girl/"
output_dir = "data/detected_faces/"
no_faces_dir = "data/no_faces/"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

if not os.path.exists(no_faces_dir):
    os.makedirs(no_faces_dir)

face_cascade = cv2.CascadeClassifier("check_points/lbpcascade_animeface.xml")

for image_name in tqdm(os.listdir(input_dir)):
    image_path = os.path.join(input_dir, image_name)
    image = cv2.imread(image_path)
    if image is None:
        continue

    faces = detect_faces(image, face_cascade)

    if len(faces) > 0:
        save_faces(image, faces, output_dir, image_name)
    else:
        # Save the original image to the no_faces directory
        cv2.imwrite(os.path.join(no_faces_dir, image_name), image)
