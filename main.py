
from pathlib import Path
import numpy as np
import cv2 as cv
import mediapipe as mp
import random
from pick import pick

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

def detect_faces(images:list[np.ndarray]):
    faces = []
    with mp_face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.5) as face_detection:
        for idx, image in enumerate(images):
            # Convert the BGR image to RGB and process it with MediaPipe Face Detection.
            results = face_detection.process(cv.cvtColor(image, cv.COLOR_BGR2RGB))

            # Draw face detections of each face.
            if not results.detections:
                print('No face detected')
                continue
            for detection in results.detections:
                if detection.score[0] < 0.75:
                    print('Face detection confidence is too low')
                    continue
                # Get the bounding box as x,y,w,h
                bbox = detection.location_data.relative_bounding_box
                h, w, _ = image.shape
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                w = int(bbox.width * w)
                h = int(bbox.height * h)

                # CUSTOM CODE
                # The bounding box only start from the middle of the forehead
                # so we need to add some pixels to the top
                y -= 50 if y - 50 > 0 else 0

                # Expand 20% on each side
                x -= int(w * 0.2)
                y -= int(h * 0.2)
                w += int(w * 0.4)
                h += int(h * 0.4)

                # Crop the face
                face = image[y:y+h, x:x+w]
                faces.append(face)

    return faces

def ordinary_points(n):
    """ordinary points are just pairs (x, y) where x and y
    are both between 0 and n - 1"""
    return [(x, y) for x in range(n) for y in range(n)]
    
def points_at_infinity(n):
    """infinite points are just the numbers 0 to n - 1
    (corresponding to the infinity where lines with that slope meet)
    and infinity infinity (where vertical lines meet)"""
    return list(range(n)) + [u"∞"]

def all_points(n):
    return ordinary_points(n) + points_at_infinity(n)

def ordinary_line(m, b, n):
    """returns the ordinary line through (0, b) with slope m
    in the finite projective plan of degree n
    includes 'infinity m'"""
    return [(x, (m * x + b) % n) for x in range(n)] + [m]

def vertical_line(x, n):
    """returns the vertical line with the specified x-coordinate
    in the finite projective plane of degree n
    includes 'infinity infinity'"""
    return [(x, y) for y in range(n)] + [u"∞"]
    
def line_at_infinity(n):
    """the line at infinity just contains the points at infinity"""
    return points_at_infinity(n)

def all_lines(n):
    return ([ordinary_line(m, b, n) for m in range(n) for b in range(n)] +
            [vertical_line(x, n) for x in range(n)] +
            [line_at_infinity(n)])
def gen_circles(number):
    for i in range(1, number + 1):
        img = np.zeros((500, 500, 3), np.uint8)
        img = cv.bitwise_not(img)
        cv.circle(img, (250, 250), 250, (0, 0, 0), -1)
        cv.putText(img, str(i), (240, 240), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        cv.imwrite("circles/" + str(i) + " " "gen.png", img)


def make_alpha(img_bgr):
    h, w, c = img_bgr.shape
    img_bgra = np.concatenate([img_bgr, np.full((h, w, 1), 255, dtype=np.uint8)], axis=-1)
    white = np.all(img_bgr == [255, 255, 255], axis=-1)
    img_bgra[white, -1] = 0
    return img_bgra


def white_to_alpha(img):
    white = np.all(img == [255, 255, 255, 255], axis=-1)
    img[white, -1] = 0
    return img

def random_rotate_in_place(image):
    angle = random.randint(0, 359)
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv.INTER_LINEAR)
    return result

def resize_image(img,size):
    return cv.resize(img, (size, size))


def put4ChannelImageOn4ChannelImage(back, fore, x, y):
    rows, cols, channels = fore.shape
    trans_indices = fore[..., 3] != 0  # Where not transparent
    overlay_copy = back[y:y + rows, x:x + cols]
    overlay_copy[trans_indices] = fore[trans_indices]
    back[y:y + rows, x:x + cols] = overlay_copy

def place_image(card, img, face_size, position):
    img = resize_image(img,face_size)
    img = random_rotate_in_place(img)
    px_center = position
    offset = int(face_size // 2)
    x = px_center[0] - offset
    y = px_center[1] - offset
    put4ChannelImageOn4ChannelImage(card, img, x, y)

def make_deck(n, pics):
    points = all_points(n)

    # create a mapping from point to pic
    mapping = { point : pic 
                for point, pic in zip(points, pics) }

    # and return the remapped cards
    return [map(mapping.get, line) for line in all_lines(n)]

def test_deck(deck):
    for c in deck:
        for f in c:
            if f is None:
                return False
    return True

def deck_options(faces):
    face_per_deck_options = []
    for i in range(3,# Min Face Per Card
                   9+1,# Max Face Per Card
                ):
        if test_deck(make_deck(i, faces)):
            face_per_deck_options.append(i)
    return face_per_deck_options
        
def faces_to_card(faces):
    # Settings
    size = 1200
    face_size = 280

    # Make the card
    card_img = np.zeros((size, size, 3), np.uint8)
    card_img = cv.bitwise_not(card_img)
    card_img = cv.cvtColor(card_img, cv.COLOR_BGR2BGRA)
    
    cv.circle(card_img, (size // 2, size // 2), 590, (0, 0, 0, 255), 10)
    faces = list(faces)
    # We want to layout these pictures in a circle
    # we need a function that takes a number of pictures, size of picture, and size of card and returns the positions of the pictures
    def layout_circle(n, face_size, card_size):
        # We want to lay out the pictures in a circle
        # We need to know the radius of the circle
        # We need to know the center of the circle
        # We need to know the angle between each picture
        # We need to know the position of each picture

        # Radius of the circle
        radius = (card_size - face_size) // 2

        # Center of the circle
        center = (card_size // 2, card_size // 2)

        # Angle between each picture
        angle = 360 / n

        # Positions of each picture
        positions = []
        for i in range(n):
            x = center[0] + radius * np.cos(np.deg2rad(angle * i))
            y = center[1] + radius * np.sin(np.deg2rad(angle * i))
            positions.append((int(x), int(y)))

        return positions
    positions = layout_circle(len(faces), face_size, size)

    for i, img in enumerate(list(faces)):
        print(f'Placing face {i}')
        p = positions[i]
        print(f'Position: {p}')
        print('Shape: ', img.shape)
        place_image(card_img, img, face_size, p)
    
    return card_img

    

def spot_it(filepaths = None):
    # Load images
    if filepaths is None:
        from tkinter import Tk
        from tkinter.filedialog import askopenfilenames
        Tk().withdraw()
        filenames = askopenfilenames()
        if not filenames:
            return
        filepaths = [Path(f) for f in filenames]
    elif len(filepaths) == 1:
        filepaths = list(Path(filepaths[0]).glob("*"))
    else:
        filepaths = [Path(f) for f in filepaths]
    for f in filepaths:
        if not f.suffix in [".jpg", ".png",'jpeg']:
            print(f"File {f} is not an image")
            return

    # Load images
    images = [cv.imread(str(f)) for f in filepaths]

    # Load the faces from every image
    faces = detect_faces(images)
    faces = [i for i in faces if i is not None and i.shape[0] > 0 and i.shape[1] > 0]
    faces = [cv.cvtColor(i, cv.COLOR_BGR2BGRA) for i in faces]
    if not faces:
        print("No faces detected")
        return
    print('Using {} faces'.format(len(faces)))

    # Shuffle the faces
    random.shuffle(faces)

    # See how many faces per card we can use
    face_per_card  = deck_options(faces)
    if not face_per_card:
        print("Invalid number of faces")
        return
    fpc = pick(face_per_card, "How many faces per card")[0] if len(face_per_card) > 1 else face_per_card[0]
    
    # Make the Deck
    deck = make_deck(fpc,faces)
    cards = []
    for d in deck:
        cards.append(faces_to_card(d))
    
    # Save the cards
    for i, card in enumerate(cards):
        cv.imwrite(f"card{i}.png", card)

        
    

    

if __name__ == "__main__":
    spot_it(filepaths = [Path(r"C:\Users\Moshe Dicker\Documents\DickerSystems\spotitgen\test")])