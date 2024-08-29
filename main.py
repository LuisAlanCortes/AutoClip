import numpy as np
import torch
import easyocr  # OCR
from PIL import Image  # OCR
import moviepy.editor as mpy
import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import cv2
import multiprocessing
import os
import sys
def main():
    debug = True
    time_stamps = []
    search_x, search_y = 400, 316 # Search box position
    search_width, search_height = 197, 128 # Search box width and height
    reader = easyocr.Reader(['en'])

    cv2_capture = cv2.VideoCapture("C:/Users/Cortes/PycharmProjects/AutoClip/Destiny Iron Banner Destruction.mp4")
    if not cv2_capture.isOpened(): # Error check for video
        print("Could not open video file")
        exit()

    frame_count = 950 # Set starting frame
    cv2_capture.set(1, frame_count)

    frame_interval = 5
    # Loop through video, reading each frame at set interval, preprocess then read set search box with OCR
    while True:
        if frame_count % frame_interval == 0:
            frame_grabbed, frame = cv2_capture.read() # Frame
            if not frame_grabbed:  # If failure to capture frame, freak out
                print("Frame empty, possible error")
                break
            if debug: # Shows frame count & frame
                print(frame_count)
                cv2.imshow('DEBUG: Original Frame', frame)
                cv2.waitKey(0)
            frame = frame[search_y:search_y+search_height, search_x:search_x+search_width] # Crop to search area
            if debug:
                cv2.imshow('DEBUG: Search Area', frame)
                cv2.waitKey(0)
            # Preprocessing for EasyOCR
            frame = cv2.resize(frame, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_CUBIC)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            kernel = np.ones((1, 1), np.uint8)
            frame = cv2.dilate(frame, kernel, iterations=1)
            frame = cv2.erode(frame, kernel, iterations=1)
            # cv2.threshold(cv2.bilateralFilter(frame, 5, 75, 75), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            cv2.threshold(cv2.medianBlur(frame, 3), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

            if debug: # Show search area after preprocessing
                cv2.imshow('DEBUG: Search Area PostFX', frame)
                cv2.waitKey(0)

            result = reader.readtext(frame)
            for (bbox, text, prob) in result:
                if "K" in text:
                    time_stamps = frame_count
                    print("Timestamp found at %i" % frame_count)
                print(f'Text: {text}, Probability: {prob}')
        frame_count += 1

    # time to frame: hours * 3600 * frame_rate + minutes * 60 * frame_rate + seconds * frame_rate
    cv2_capture.release()



def cut_video(markers, video, prefix, suffix):
    video_object = mpy.VideoFileClip(video)
    clips = [0]
    for i in range(len(markers)):
        clips[i] = video_object.subclip((markers[i]-prefix), (markers[i]+suffix))
    final_video = mpy.concatenate_videoclips(clips)


if __name__ == '__main__':
    main()