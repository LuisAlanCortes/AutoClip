import moviepy.editor as mpy
import cv2
import os

# Test File "D:\Footage\2023-12-03 22-55-09.mp4"

footagedirectory = "D:\\Footage\\"
title = "D:\\Footage\\2023-12-03 22-55-09.mp4"
savetitle = title.replace(footagedirectory, "")

# Test Edit points ('00:00:02.949', '00:00:04.152'), ('00:00:06.328', '00:00:13.077')
editpoints = [
    ('00:00:02.949', '00:00:04.152'),
    ('00:00:06.328', '00:00:13.077')
]

def find_edit_points():
    


def cut_video(savetitle, editpoints):
    video = mpy.VideoFileClip(title)

    clips = []
    for editpoint in editpoints:
        clip = video.subclip(editpoint[0], editpoint[1])
        clips.append(clip)

    final_clip = mpy.concatenate_videoclips(clips)

    final_clip.write_videofile(savetitle)
    video.close()

if __name__ == '__main__':
    cut_video(savetitle, editpoints)