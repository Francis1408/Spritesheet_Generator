import cv2 as cv
from pathlib import Path
import numpy as np
import sys
from config import SPRITE_POS

DATASET_PATH = Path("../../Dataset/train")
OUTPUT_PATH = Path("../../Dataset/spriteSheet")
SPRITE_SIZE = (512, 512)
NEW_DIMENSIONS = (256, 256)


def showProgress(current, total, message=None, bar_length = 45):

    if message:
        sys.stdout.write(f"\n{message}\n")

    percent = current / total
    arrow = '█' * int(percent * bar_length)
    spaces = ' ' * (bar_length - len(arrow))

    sys.stdout.write(f"\rProgress: [{arrow}{spaces}] {percent*100:.2f}%")
    sys.stdout.flush()

    if current == total:
        sys.stdout.write('\n')


def scaleImage(image):
    return cv.resize(image, NEW_DIMENSIONS, interpolation=cv.INTER_NEAREST)

def imageOverlay(pos, img, bg_image):

    # Get the position coordinates
    coordinates = SPRITE_POS[pos]

    # Read image path
    fg_image = cv.imdecode(np.frombuffer(img, np.uint8), cv.IMREAD_UNCHANGED)

    fg_image = scaleImage(fg_image) # Scale up the image
    fg_heigth, fg_width, c = fg_image.shape
   
    x, y = coordinates
    bg_image[y:y+fg_heigth, x:x+fg_width] = fg_image

    return bg_image

def build_sprite_sheet(position_images):

    
    try:

        # Create a blank background
        sprite = np.zeros((*SPRITE_SIZE, 4), dtype=np.uint8)

        for pos, img in position_images.items():
            
            # Build the sprite sheet
            if img:
                sprite = imageOverlay(pos, img, sprite)

        return sprite

    except Exception as e:
        raise f"Error : Could not build sprite sheet: {e}."
        
   


