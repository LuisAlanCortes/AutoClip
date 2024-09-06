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
    if debug: # OCR Testing
        marker_coordinates = (511, 306, 70, 103)
        frame_fig_size = (0.7, 1.03)
        marker_text = "ki"
        time_stamps = process_video_ocr(debug, marker_coordinates, marker_text, frame_fig_size)
    else:
        marker_coordinates, marker_frame, frame_fig_size = get_marker_location()  # Get marker location, save frame & coordinates
        marker = get_marker(marker_frame)  # Get marker image
        marker_text = get_marker_text() # Ask for marker text
        if marker_text:
            time_stamps = process_video_ocr(debug, marker_coordinates, marker_text, frame_fig_size) # OCR
        else:
            time_stamps = process_video_tm(debug, marker_coordinates, marker, frame_fig_size) # Template Matching
    print(time_stamps)


def process_video_tm(debug, marker_coordinates, marker, fig_size):
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
        frame_count += 1
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

                plot(1, 2, 1, 'Original Image', original) # Original Image

                # Highlight search area in the original image
                x_start, y_start, w, h = marker_coordinates
                cv2.rectangle(original, (x_start, y_start), (x_start + w, y_start + h), (255, 0, 0), 2)

                # Draw the rectangle around the detected template location
                cv2.rectangle(original, top_left, (top_left[0] + marker.shape[1], top_left[1] + marker.shape[0]),
                              (0, 255, 0), 2)

                plot(1, 2, 2, 'Matching Result', result) # Template Matching Result
                plt.tight_layout()
                plt.show()

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
        exit()

    marker_rect = cv2.selectROI("Select marker location", frame, False)  # Select marker zone
    marker_crop = frame[int(marker_rect[1]):int(marker_rect[1] + marker_rect[3]),
                  int(marker_rect[0]):int(marker_rect[0] + marker_rect[2])]  # Crop frame
    marker_coordinates = marker_rect[0], marker_rect[1], marker_rect[2], marker_rect[3] # Store zone coordinates
    height, width, _ = marker_crop.shape
    frame_fig_size = (width / 100, height / 100)
    cv2_capture.release()
    cv2.destroyAllWindows()
    return marker_coordinates, marker_timestamp, frame_fig_size


def get_marker(marker_frame):
    cv2_capture = cv2.VideoCapture("C:/Users/Cortes/PycharmProjects/AutoClip/Destiny Iron Banner Destruction.mp4")
    if not cv2_capture.isOpened():  # Error check for video
        print("Could not open video file")
        exit()
    cv2_capture.set(1, marker_frame)
    frame_grabbed, frame = cv2_capture.read()  # Frame
    if not frame_grabbed:
        print("Frame empty, possible error")
        exit()

    marker_rect = cv2.selectROI("Select marker", frame, False)  # Select ROI
    marker_crop = frame[int(marker_rect[1]):int(marker_rect[1] + marker_rect[3]),
                  int(marker_rect[0]):int(marker_rect[0] + marker_rect[2])]  # Grab crop location
    marker = preprocess_tm(marker_crop, True)
    cv2_capture.release()
    cv2.destroyAllWindows()
    return marker


def get_marker_text():
    return input("Input search text for OCR, leave blank for template matching")


def preprocess_tm(frame, debug):
    if debug:
        plt.figure(figsize=(12, 8))
        plot(3,2,1,'Original Frame', frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if debug:
        plot(3,2,2,'Grayscale', frame)
    frame = cv2.GaussianBlur(frame, (5, 5), 0)
    if debug:
        plot(3,2,3,'Gaussian Blur', frame)
    frame = cv2.equalizeHist(frame)
    if debug:
        plot(3,2,4,'Histogram equal', frame)
    frame = cv2.Canny(frame, 50, 150)
    if debug:
        plot(3,2,5,'Canny', frame)
    _, frame = cv2.threshold(frame, 127, 255, cv2.THRESH_BINARY)
    if debug:
        plot(3,2,6,'Threshold', frame)
        plt.tight_layout()
        plt.show()
    return frame


def plot(row, col, index, text, frame):
    plt.subplot(row, col, index)
    plt.title(text)
    plt.imshow(frame, cmap='gray')
    plt.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)

def preprocess_ocr(frame, debug):
    # if debug:
    #     plt.figure(figsize=(12, 8))
    #     plot(3,3,1,'Original Frame', frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # if debug:
    #     plot(3,3,2,'Grayscale', frame)
    frame = cv2.bitwise_not(frame)
    # if debug:
    #     plot(3,3,3,'Invert', frame)
    frame = cv2.adaptiveThreshold(frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    # if debug:
    #     plot(3,3,4,'Adaptive Thresholding', frame)
    # frame = cv2.medianBlur(frame, 3)
    # if debug:
    #     plot(3,3,5,'Denoise', frame)
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # frame = cv2.dilate(frame, kernel, iterations=1)
    # if debug:
    #     plot(3,3,6,'Dilation', frame)
    # frame = cv2.erode(frame, kernel, iterations=1)
    # if debug:
    #     plot(3,3,7,'Threshold', frame)
    #     plt.tight_layout()
    #     plt.show()
    return frame


def process_video_ocr(debug, marker_coordinates, marker_text, fig_size):
    time_stamps = []
    reader = easyocr.Reader(['en'])

    cv2_capture = cv2.VideoCapture("C:/Users/Cortes/PycharmProjects/AutoClip/Destiny Iron Banner Destruction.mp4")
    if not cv2_capture.isOpened(): # Error check for video
        print("Could not open video file")
        exit()

    frame_total = int(cv2_capture.get(cv2.CAP_PROP_FRAME_COUNT)) # Get frame total
    frame_count = 0 # Frame counter
    frame_interval = 20 # Interval to skip frames
    marker_interval = 90 # Duration of marker
    frames_read = 0 # Frames read
    marker_found = False
    cv2_capture.set(1, frame_count) # Set video position

    # Loop through video, reading each frame at set interval, preprocess then read set search box with OCR
    while frames_read < frame_total:
        if marker_found:
            frame_count = frame_count + marker_interval
            cv2_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
            marker_found = False
        frame_count += 1
        if frame_count % frame_interval == 0:
            frames_read +=1
            cv2_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
            frame_grabbed, frame = cv2_capture.read() # Frame
            if not frame_grabbed:
                print("Frame empty, possible error")
                break
            if debug: # Shows frame counts
                print(f"Frame Count: {frame_count}, {frames_read}")
                original = frame
            frame = frame[int(marker_coordinates[1]):int(marker_coordinates[1] + marker_coordinates[3]),
                    int(marker_coordinates[0]):int(marker_coordinates[0] + marker_coordinates[2])]  # Crop to marker
            frame = preprocess_ocr(frame, debug)

            results = reader.readtext(frame) # Read frame with Easy OCR
            for result in results:
                bbox, text, prob = result
                if marker_text.lower() in text.lower():
                    time_stamps.append(frame_count) # Save frame count to time stamp list
                    marker_found = True
                    # # Skip forward marker interval
                    # new_frame_position = frame_count + marker_interval
                    # cv2_capture.set(cv2.CAP_PROP_POS_FRAMES, new_frame_position)

                    if debug:
                        print(f"Timestamp found at {frame_count}")
                        print(f'Text: {text}, Probability: {prob}')
                        plt.figure(figsize=(8,5))
                        plot(1, 1, 1, 'Original Image', original)  # Original Image
                        plt.tight_layout()
                        plt.show()

    # time to frame: hours * 3600 * frame_rate + minutes * 60 * frame_rate + seconds * frame_rate
    cv2_capture.release()
    cv2.destroyAllWindows()
    return time_stamps


if __name__ == '__main__':
    main()