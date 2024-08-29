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


def process_video(video_path, target_text):
    capture = cv2.VideoCapture(video_path)
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))  # Get total frame count
    capture.release()
    chunk_size = 1000
    frame_chunks = [[i, i + chunk_size] for i in range(0, total, chunk_size)]  # Split frames into chunk lists
    frame_chunks[-1][-1] = min(frame_chunks[-1][-1], total)

    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        futures = [executor.submit(process_frame, target_text, f[0], f[1]) for f in frame_chunks]


def process_frame(target, start, end):
    capture = cv2.VideoCapture(target)
    reader = easyocr.Reader(['en'])

    if start < 0:  # if start isn't specified lets assume 0
        start = 0
    if end < 0:  # if end isn't specified assume the end of the video
        end = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    capture.set(1, start)  # set the starting frame of the capture
    frame = start
    while frame < end:
        frame_grabbed, frame = capture.read()
        cv2.imshow('frame', frame)
        if not frame_grabbed:
            print("Frame empty, possible error")
        result = reader.readtext(frame)
        if target in result:
            print("Target found at {}".format(str(datetime.timedelta(
                milliseconds=capture.get(cv2.CAP_PROP_POS_MSEC)))[2:]))


# Take time markers and cut video at the marker - the prefix amount and + the suffix
def cut_video(markers, video, prefix, suffix):
    video_object = mpy.VideoFileClip(video)
    clips = [0]
    for i in range(len(markers)):
        clips[i] = video_object.subclip((markers[i]-prefix), (markers[i]+suffix))
    final_video = mpy.concatenate_videoclips(clips)


def slow_process_video(video_path, target_text):
    reader = easyocr.Reader(['en'])
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Could not open video file")
        exit()

    # Loop until video is over, sending each frame to processing, and saving its time stamp
    while True:
        frame_grabbed, frame = cap.read()
        if not frame_grabbed:  # If failure to capture frame, freak out
            break
        else:  # Process frame
            if slow_process_text(frame, target_text, reader):
                # Save time in hr:mn:sc to array, chop off date with [2:]
                print("Target found at {}".format(str(datetime.timedelta(
                    milliseconds=cap.get(cv2.CAP_PROP_POS_MSEC)))[2:]))

    # time to frame: hours * 3600 * frame_rate + minutes * 60 * frame_rate + seconds * frame_rate
    cap.release()


def slow_process_text(search_img, search_text, reader):
    result = reader.readtext(search_img)
    if search_text in result:
        return True


def main():
    main()


if __name__ == '__main__':
    user_target_img = "C:/Users/Cortes/PycharmProjects/AutoClip/AutoClipTestImg.png"
    user_video = "C:/Users/Cortes/PycharmProjects/AutoClip/AutoClipTestFootage.mp4"
    user_video_2 = "C:/Users/Cortes/PycharmProjects/AutoClip/Destiny Iron Banner Destruction.mp4"
    user_target_text = "Kill"
    marker_timestamps = ['00:00:00']

    start_time = datetime.datetime.now()
    #process_video(user_video, user_target_text)
    slow_process_video(user_video, user_target_text)
    #frames = video_search(user_target_text, user_video, marker_timestamps)
    end_time = datetime.datetime.now()

    print("\nTime: {:.2f}".format((end_time - start_time).total_seconds()))
    #print("FPS: {:.2f}\n".format(frames / (end_time - start_time).total_seconds()))

    #for marker_timestamp in marker_timestamps:
        #print(marker_timestamp)
