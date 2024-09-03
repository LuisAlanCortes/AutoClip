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
import matplotlib.pyplot as plt


def main():
    debug = True  # Show all frames being processed, and sets the read to first kill
    marker_coordinates, marker_frame = get_marker_location() # Get marker location, save frame & coordinates
    marker = get_marker(marker_frame) # Get marker image
    if debug:
        time_stamps = process_video_tm(debug, marker_coordinates, marker) # Template Matching
    else:
        marker_text = get_marker_text() # Ask for marker text
        if marker_text:
            time_stamps = process_video_ocr(debug, marker_coordinates, marker_text) # OCR
        else:
            time_stamps = process_video_tm(debug, marker_coordinates, marker) # Template Matching
    print(time_stamps)


def process_video_tm(debug, marker_coordinates, marker):
    time_stamps = []
    cv2_capture = cv2.VideoCapture("C:/Users/Cortes/PycharmProjects/AutoClip/Destiny Iron Banner Destruction.mp4")
    if not cv2_capture.isOpened():  # Error check for video
        print("Could not open video file")
        exit()
    frame_count = 0
    if debug:
        frame_count = 1000

    cv2_capture.set(1, frame_count) # Set starting frame

    frame_interval = 20 # Interval between frames
    frames_read = 0
    # Loop through video, reading each frame at interval, preprocess, then template match
    while True:
        if frame_count % frame_interval == 0: # Interval
            frames_read +=1
            frame_grabbed, frame = cv2_capture.read()
            if not frame_grabbed:
                print("Frame empty, possible error")
                break
            if debug:  # Show frame & count
                print("Frame Count: %i" % frame_count)
                print("Frames Read: %i" % frames_read)
                original = frame

            frame = frame[int(marker_coordinates[1]):int(marker_coordinates[1] + marker_coordinates[3]),
                  int(marker_coordinates[0]):int(marker_coordinates[0] + marker_coordinates[2])]  # Crop to marker
            frame = preprocess_tm(frame, debug)

            result = cv2.matchTemplate(frame, marker, cv2.TM_CCOEFF_NORMED) # Template match
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result) # Match Location
            top_left = max_loc # Match coordinate

            if debug: # Show original, crop zone, and match
                plt.figure(figsize=(12, 8))

                # Original Image
                plt.subplot(1, 2, 1)
                plt.title('Original Image')
                plt.imshow(original)
                plt.axis('off')

                # Highlight search area in the original image
                x_start, y_start, w, h = marker_coordinates
                cv2.rectangle(original, (x_start, y_start), (x_start + w, y_start + h), (255, 0, 0), 2)

                # Draw the rectangle around the detected template location
                cv2.rectangle(original, top_left, (top_left[0] + marker.shape[1], top_left[1] + marker.shape[0]),
                              (0, 255, 0), 2)

                # Template Matching Result
                plt.subplot(1, 2, 2)
                plt.title('Matching Result')
                plt.imshow(result, cmap='hot')
                plt.axis('off')

                plt.tight_layout()
                plt.show()
            # Template Matching with all methods
            # # Template matching methods
            # methods = [
            #     cv2.TM_CCOEFF,
            #     cv2.TM_CCOEFF_NORMED,
            #     cv2.TM_CCORR,
            #     cv2.TM_CCORR_NORMED,
            #     cv2.TM_SQDIFF,
            #     cv2.TM_SQDIFF_NORMED
            # ]
            # method_names = [
            #     'TM_CCOEFF',
            #     'TM_CCOEFF_NORMED',
            #     'TM_CCORR',
            #     'TM_CCORR_NORMED',
            #     'TM_SQDIFF',
            #     'TM_SQDIFF_NORMED'
            # ]
            #
            # results = []
            # for method in methods:
            #     # Apply template matching
            #     result = cv2.matchTemplate(marker, frame, method)
            #
            #     # Normalize result for visualization
            #     if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            #         min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            #         result_normalized = (result - min_val) / (max_val - min_val)
            #     else:
            #         result_normalized = cv2.normalize(result, None, 0, 1, cv2.NORM_MINMAX)
            #
            #     results.append(result_normalized)
            #
            # plt.figure(figsize=(15, 12))
            #
            # # Original Frame
            # plt.subplot(3, 3, 1)
            # plt.title('Original Frame')
            # plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            # plt.axis('off')
            #
            # # Template
            # plt.subplot(3, 3, 2)
            # plt.title('Template')
            # plt.imshow(cv2.cvtColor(marker, cv2.COLOR_BGR2RGB))
            # plt.axis('off')
            #
            # # Template matching results
            # for i, (result, method_name) in enumerate(zip(results, method_names), start=3):
            #     plt.subplot(3, 3, i)
            #     plt.title(method_name)
            #     plt.imshow(result, cmap='hot')
            #     plt.axis('off')
            #
            # plt.tight_layout()
            # plt.show()

            #print("Timestamp found at %i" % frame_count)
        frame_count += 1

    # time to frame: hours * 3600 * frame_rate + minutes * 60 * frame_rate + seconds * frame_rate
    cv2_capture.release()
    cv2.destroyAllWindows()
    return time_stamps


def cut_video(markers, video, prefix, suffix):
    video_object = mpy.VideoFileClip(video)
    clips = [0]
    for i in range(len(markers)):
        clips[i] = video_object.subclip((markers[i]-prefix), (markers[i]+suffix))
    final_video = mpy.concatenate_videoclips(clips)


def get_marker_location():
    marker_timestamp = input("Input timestamp where search marker will show(e.g. 02:25): ")
    seconds = 0 # Convert timestamp to seconds
    for block in marker_timestamp.split(':'):
        seconds = seconds * 60 + int(block, 10)
    marker_timestamp = seconds

    # Open video and set it to frame given by user
    cv2_capture = cv2.VideoCapture("C:/Users/Cortes/PycharmProjects/AutoClip/Destiny Iron Banner Destruction.mp4")
    if not cv2_capture.isOpened():
        print("Could not open video file")
        exit()
    marker_fps = cv2_capture.get(cv2.CAP_PROP_FPS)
    marker_timestamp = float(marker_timestamp) * marker_fps # Calculate frame based on second
    cv2_capture.set(1, marker_timestamp)
    frame_grabbed, frame = cv2_capture.read()
    if not frame_grabbed:
        print("Frame empty, possible error")

    print("Select marker location")
    marker_rect = cv2.selectROI(frame, False)  # Select marker zone
    marker_crop = frame[int(marker_rect[1]):int(marker_rect[1] + marker_rect[3]),
                  int(marker_rect[0]):int(marker_rect[0] + marker_rect[2])]  # Crop frame
    marker_coordinates = marker_rect[0], marker_rect[1], marker_rect[2], marker_rect[3] # Store zone coordinates

    cv2.imshow("Marker", marker_crop)  # Display cropped image
    cv2.waitKey(0)

    cv2_capture.release()
    cv2.destroyAllWindows()
    return marker_coordinates, marker_timestamp


def get_marker(marker_frame):
    cv2_capture = cv2.VideoCapture("C:/Users/Cortes/PycharmProjects/AutoClip/Destiny Iron Banner Destruction.mp4")
    if not cv2_capture.isOpened():  # Error check for video
        print("Could not open video file")
        exit()
    cv2_capture.set(1, marker_frame)
    frame_grabbed, frame = cv2_capture.read()  # Frame
    print("Select marker")
    marker_rect = cv2.selectROI(frame, False)  # Select ROI
    marker_crop = frame[int(marker_rect[1]):int(marker_rect[1] + marker_rect[3]),
                  int(marker_rect[0]):int(marker_rect[0] + marker_rect[2])]  # Grab crop location
    marker_coordinates = marker_rect[0], marker_rect[1], marker_rect[2], marker_rect[3]
    marker = preprocess_tm(marker_crop, True)
    cv2_capture.release()
    cv2.destroyAllWindows()
    return marker


def get_marker_text():
    return input("Input search text for OCR, leave blank for template matching")


def preprocess_tm(frame, debug):
    if debug:
        plt.figure(figsize=(12, 8))
        plt.subplot(3, 2, 1)
        plt.title('Original Frame')
        plt.imshow(frame)
        plt.axis('off')
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if debug:
        plt.subplot(3, 2, 2)
        plt.title('Grayscale')
        plt.imshow(frame, cmap='gray')
        plt.axis('off')
    frame = cv2.GaussianBlur(frame, (5, 5), 0)
    if debug:
        plt.subplot(3, 2, 3)
        plt.title('Gaussian Blur')
        plt.imshow(frame, cmap='gray')
        plt.axis('off')
    frame = cv2.equalizeHist(frame)
    if debug:
        plt.subplot(3, 2, 4)
        plt.title('Histogram equal')
        plt.imshow(frame, cmap='gray')
        plt.axis('off')
    frame = cv2.Canny(frame, 50, 150)
    if debug:
        plt.subplot(3, 2, 5)
        plt.title('Canny')
        plt.imshow(frame, cmap='gray')
        plt.axis('off')
    _, frame = cv2.threshold(frame, 127, 255, cv2.THRESH_BINARY)
    if debug:
        plt.subplot(3, 2, 6)
        plt.title('Threshold')
        plt.imshow(frame, cmap='gray')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    return frame


def preprocess_ocr(frame, debug):
    if debug:
        plt.figure(figsize=(12, 8))
        plt.subplot(3, 3, 1)
        plt.title('Original Frame')
        plt.imshow(frame)
        plt.axis('off')
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if debug:
        plt.subplot(3, 3, 2)
        plt.title('Grayscale')
        plt.imshow(frame, cmap='gray')
        plt.axis('off')
    frame = cv2.bitwise_not(frame)
    if debug:
        plt.subplot(3, 3, 3)
        plt.title('Invert')
        plt.imshow(frame, cmap='gray')
        plt.axis('off')
    frame = cv2.adaptiveThreshold(frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    if debug:
        plt.subplot(3, 3, 4)
        plt.title('Adaptive Thresholding')
        plt.imshow(frame, cmap='gray')
        plt.axis('off')
    frame = cv2.medianBlur(frame, 3)
    if debug:
        plt.subplot(3, 3, 5)
        plt.title('Denoise')
        plt.imshow(frame, cmap='gray')
        plt.axis('off')
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    frame = cv2.dilate(frame, kernel, iterations=1)
    if debug:
        plt.subplot(3, 3, 6)
        plt.title('Dilation')
        plt.imshow(frame, cmap='gray')
        plt.axis('off')
    eroded = cv2.erode(frame, kernel, iterations=1)
    if debug:
        plt.subplot(3, 3, 7)
        plt.title('Threshold')
        plt.imshow(frame, cmap='gray')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    return frame


def process_video_ocr(debug, marker_coordinates, marker_text):
    time_stamps = []
    # search_x, search_y = 400, 316 # Search box position
    # search_width, search_height = 197, 128 # Search box width and height
    reader = easyocr.Reader(['en'])

    cv2_capture = cv2.VideoCapture("C:/Users/Cortes/PycharmProjects/AutoClip/Destiny Iron Banner Destruction.mp4")
    if not cv2_capture.isOpened(): # Error check for video
        print("Could not open video file")
        exit()
    frame_count = 0
    if debug:
        frame_count = 950 # Set starting frame
    else:
        frame_count = 0
    cv2_capture.set(1, frame_count)

    frame_interval = 20
    frames_read = 0
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
            frame = frame[int(marker_coordinates[1]):int(marker_coordinates[1] + marker_coordinates[3]),
                    int(marker_coordinates[0]):int(marker_coordinates[0] + marker_coordinates[2])]  # Crop to marker
            if debug:
                cv2.imshow('DEBUG: Search Area', frame)
                cv2.waitKey(0)
            # Preprocessing for EasyOCR
            frame = preprocess_ocr(frame, debug)

            if debug: # Show search area after preprocessing
                cv2.imshow('DEBUG: Search Area PostFX', frame)
                cv2.waitKey(0)

            search_term = "K"
            result = reader.readtext(frame) # Read frame with Easy OCR
            for (bbox, text, prob) in result: # Search for search term in text
                if search_term in text:
                    time_stamps.append(frame_count) # Save frame count to time stamp list
                    #if debug: # Print frame found
                    print("Timestamp found at %i" % frame_count)
                if debug: # Print all text found
                    print(f'Text: {text}, Probability: {prob}')
            # frames_read +=1
            # print(frames_read)
        frame_count += 1

    # time to frame: hours * 3600 * frame_rate + minutes * 60 * frame_rate + seconds * frame_rate
    cv2_capture.release()
    cv2.destroyAllWindows()
    return time_stamps


if __name__ == '__main__':
    main()