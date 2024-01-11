import moviepy.editor as mpy
from matplotlib import pyplot as plt
import cv2
import os
from random import randint

# Test File "D:\Footage\2023-12-03 22-55-09.mp4"


def test_video(title):
    cap = cv2.VideoCapture(title)
    while True:
        success, img = cap.read()
        cv2.imshow("Video", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()


# def find_edit_points(title):

# Cut video based on given points and stitch it back together
def cut_video(title, edit_points):
    video = mpy.VideoFileClip(title)

    clips = []
    for edit_point in edit_points:
        clip = video.subclip(edit_point[0], edit_point[1])
        clips.append(clip)

    final_clip = mpy.concatenate_videoclips(clips)

    final_clip.write_videofile(save_title)
    video.close()


def main():
    main()


if __name__ == '__main__':
    footage_directory = 'D:\\Footage\\'
    original_title = 'D:\\Footage\\2023-12-03 22-55-09.mp4'
    save_title = footage_directory, 'AutoClip', randint(1000, 2000)

    # Array of start and end points for editing | Test('00:00:02.949', '00:00:04.152'), ('00:00:06.328', '00:00:13.077')
    edit_points = [
        ('00:00:02.949', '00:00:04.152'),
        ('00:00:06.328', '00:00:13.077')
    ]
    find_edit_points(original_title)
    # cut_video(save_title, edit_points)

