import numpy as np
import cv2 as cv
from pathlib import Path
import mediapipe as mp
import random
from pick import pick
from deck import *
from card import *

faces_folder = Path.cwd() / 'faces'
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
FACE_DETECTION_EXPAND = 0.2

def file_selector():
    from tkinter import Tk
    from tkinter.filedialog import askopenfilenames
    Tk().withdraw()
    filenames = askopenfilenames()
    if not filenames:
        return
    filepaths = [Path(f) for f in filenames]
    return filepaths

def load_images(folder:Path) -> list:
    images = []
    for image_path in list(folder.glob('*.png')) + list(folder.glob('*.jpg')) + list(folder.glob('*.jpeg')):
        image = cv.imread(str(image_path))
        if image is None:
            print(f'Could not read {image_path}')
            continue
        if image.shape[2] == 4:
            image = cv.cvtColor(image, cv.COLOR_BGRA2BGR)
        images.append(image)
    return images

def img_to_faces(img:np.ndarray, n:int) -> list:
    faces = []

    with mp_face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.5) as face_detection:
        # Convert the BGR image to RGB and process it with MediaPipe Face Detection.
        results = face_detection.process(cv.cvtColor(img, cv.COLOR_BGR2RGB))
        if not results.detections:
            return []

        for detection in results.detections:

            if detection.score[0] < 0.75:
                continue

            # Get the bounding box as x,y,w,h
            bbox = detection.location_data.relative_bounding_box
            h, w, _ = img.shape
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            w = int(bbox.width * w)
            h = int(bbox.height * h)

            # Expand 20% on each side
            x -= int(w * FACE_DETECTION_EXPAND)
            y -= int(h * FACE_DETECTION_EXPAND)
            w += int(w * FACE_DETECTION_EXPAND * 2)
            h += int(h * FACE_DETECTION_EXPAND * 2)

            # Crop the face
            face = img[y:y+h, x:x+w]
            faces.append(face)

    return faces

def make_face_flow():
    input('Select images to make faces from. Press enter to continue')
    filepaths = file_selector()
    # Load all images
    images = [cv.imread(str(f)) for f in filepaths]
    images = [img for img in images if img is not None]
    images = [i for i in images if i.shape[0] > 0 and i.shape[1] > 0]
    if not images:
        print('No images selected')
        return
    faces = []
    # Get all faces from all images
    for img in images:
        faces.extend(img_to_faces(img, 1))
    
    if not faces:
        print('No faces found')
        return
    
    # Save all faces to faces folder
    faces_folder.mkdir(exist_ok=True)
    for i, face in enumerate(faces):
        cv.imwrite(str(faces_folder / f'face{i}.png'), face)
    
    print(f'Saved {len(faces)} faces to faces folder')
    print('Go through the faces folder and delete any faces that are not good')

def main():
    if not faces_folder.exists():
        make_face_flow()

    # Check if any images in the faces folder, png or jpg
    faces = load_images(faces_folder)
    if not faces:
        make_face_flow()
    
    # Convert every face to BGRA
    faces = [cv.cvtColor(face, cv.COLOR_BGR2BGRA) for face in faces]
    print('Using {} faces from the faces folder'.format(len(faces)))
    input('Press enter to continue')

    random.shuffle(faces)
    face_per_card  = deck_options(faces)
    if not face_per_card:
        print("Invalid number of faces")
        return
    fpc = pick(face_per_card, "How many faces per card")[0] if len(face_per_card) > 1 else face_per_card[0]
    deck = make_deck(fpc,faces)
    cards = []
    for d in deck:
        cards.append(faces_to_card(d))
    # Save the cards
    
    for i, card in enumerate(cards):
        cv.imwrite(f"card{i}.png", card)


if __name__ == '__main__':
    main()




