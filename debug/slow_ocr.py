import pytesseract  # OCR
from PIL import Image  # OCR
import moviepy.editor as mpy
import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import cv2
import multiprocessing
import os
import sys


# Read in video as CV2 object, loop through each frame and send for processing
def video_search_slow(target_text, video, timestamps):
    cap = cv2.VideoCapture(video)

    if not cap.isOpened():
        print("Could not open video file")
        exit()
    # frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    # frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # fps = int(cap.get(cv2.CAP_PROP_FPS))
    # cap.set(cv2.CAP_PROP_POS_MSEC, frame_rate)
    current_frame = 0

    # Loop until video is over, sending each frame to processing, and saving its time stamp
    while True:
        current_frame += 1
        frame_grabbed, frame = cap.read()
        if not frame_grabbed:  # If failure to capture frame, freak out
            break
        else:  # Process frame
            if process_text_slow(frame, target_text):
                # Save time in hr:mn:sc to array, chop off date with [2:]
                timestamps.append(str(datetime.timedelta(milliseconds=cap.get(cv2.CAP_PROP_POS_MSEC)))[2:])

    # time to frame: hours * 3600 * frame_rate + minutes * 60 * frame_rate + seconds * frame_rate
    cap.release()
    cv2.destroyAllWindows()
    return current_frame


# https://pythongeeks.org/python-opencv-text-detection-and-extraction/
# search_img: CV2 object to be searched. search_text: Text to find in image
# Read cv2 object, convert to binary, search for text. Check text for search term.
def process_text_slow(search_img, search_text):
    pytesseract.pytesseract.tesseract_cmd = r"D:/Pytesseract/tesseract.exe"  # Tesseract Location

    search_img_greyscale = cv2.cvtColor(search_img, cv2.COLOR_BGR2GRAY)  # Convert frame to greyscale + binary
    _, search_img_binary = cv2.threshold(search_img_greyscale, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Get all possible text from frame
    search_img_text = pytesseract.image_to_string(Image.fromarray(search_img_binary), config='--psm 11')
    if search_text in search_img_text:
        return True
    else:
        return False

    # search_img_text = search_img_text.replace("\n")  # DEBUG: Remove new line characters for ease of view
    # cv2.imshow(search_img_text, search_img)  # DEBUG: Display search image, and search text (As win_title)