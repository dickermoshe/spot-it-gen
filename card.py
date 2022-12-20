import numpy as np
import cv2 as cv
import random


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
            # We need to make sure that the faces dont overlap or go outside the card
            
            # Get the angle of the picture
            a = angle * i
            
            # Convert the angle to radians
            r = (a * np.pi) / 180
            
            # Get the position of the picture
            x = int(center[0] + radius * np.cos(r))
            y = int(center[1] + radius * np.sin(r))

            # Add the position to the list
            positions.append((x, y))
            

        return positions
    positions = layout_circle(len(faces), face_size, size)

    for i, img in enumerate(list(faces)):
        p = positions[i]
        place_image(card_img, img, face_size, p)
    
    return card_img

    
