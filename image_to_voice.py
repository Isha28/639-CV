import cv2
import os
from pytesseract import pytesseract
import numpy as np
from PIL import Image
from TTS.TTS.bin.connector2 import tts

def invert_image(img):
    inverted_image = cv2.bitwise_not(img)
    return inverted_image
    
def convert_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def binarization(gray_image):
    thresh, im_bw = cv2.threshold(gray_image, 120, 230, cv2.THRESH_BINARY)
    return im_bw

def noise_removal(image):
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.erode(image, kernel, iterations=1)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    image = cv2.medianBlur(image, 3)
    return image

def tesseract(input_image):
    path_to_tesseract = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    pytesseract.tesseract_cmd = path_to_tesseract
    gray = cv2.imread(input_image)
    custom_oem_psm_config = r'--oem 3 --psm 6'
    temp_file = "{}.jpg".format(os.getpid())
    cv2.imwrite(temp_file, gray)
    text = pytesseract.image_to_string(Image.open(temp_file), config=custom_oem_psm_config)
    os.remove(temp_file)
    return text
    
def save_img(image, file):
    cv2.imwrite(file, image)
    
def main(input_image):
    file = "check_noise.jpg"
    image = cv2.imread(input_image)
    image = convert_grayscale(image)
    image = binarization(image)
    input_image = noise_removal(image)
    save_img(input_image, file)
    text = tesseract(file)
    tts(text, "out.wav")
    return text
    
if __name__ == "__main__":
    main("sample_images/page_02.png")
