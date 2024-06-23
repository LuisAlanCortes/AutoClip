import cv2
import numpy as np
import matplotlib.pyplot as plt
import pytesseract  # OCR
from PIL import Image  # OCR
import moviepy.editor as mpy
import datetime


# Program Overview:
# User can import a jpg/png to match. Selection: Video file/multi-select of files (Mult select or folder select).
# Selection: Cut duration (default: 30 seconds before, 5 seconds after).
# Selection: match percentage.
# Selection: Save location.
# Selection: Stitch together function.
# Program looks for matches of image file in videos. Once done processing, a video player will show the files, and
# visibly show the markers of potential matches (shows percentage match), as well as the cut markers around those
# matches. User can veto markers if they do not match enough. Once user continues, program splices the cuts into a
# complete video file.

# Structure:
# Menu - File selector / file location input for jpg/png match target, mp4 video file or folder selector
#   for mp4s, save location, 0-100 number input for match percentage.
# SearchVideo - Loop through video files, search for potential matches, check match percentage against given percentage
#   threshold. If above, save time location to cut array, and save the pre-marker and post-marker times to their
#   respective arrays based on the given durations.
# VideoPlayer - Display videos with each cut marked, with cut duration highlighted around them. When selected,
#   allow user to unselect marker, creating an index marker pointer to be ignored in the stitcher.
# CutVideo - Loop through the cut array, skipping the unselected-index markers, cut out the given markers from the pre-
#   and post-marker array, save as individual clips.
# Stitcher - If selected: Stitch together clips in the target folder into one larger video from the cut video clips
#   Alert user operation complete, display stitched video using video player.

# Test File "D:\Footage\2023-12-03 22-55-09.mp4"

# userTarget = "C:/Users/Cortes/PycharmProjects/AutoClip/AutoClipTestImg.png"
# userVideo = "C:/Users/Cortes/PycharmProjects/AutoClip/Destiny Iron Banner Destruction.mp4"


# Debug - Template matching with all possible methods. Shows result of all methods and box around all matches for frames
def debug_opencv_allmethods(target, video):
    target_img = cv2.imread(target)  # Read in target image and convert to proper colors
    target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)
    target_height, target_width, _ = target_img.shape

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():  # Error check
        print("Could not open video file")
        exit()

    #frame_rate = cap.get(cv2.CAP_PROP_FPS)
    #frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    #duration = frame_count / frame_rate
    #cap.set(cv2.CAP_PROP_POS_MSEC, frame_rate)

    while cap.isOpened():  # While video capture is going
        ret, frame = cap.read()  # Capture frame of video
        if not ret:  # If failure to capture frame, freak out
            print("Could not read frame")
        else:  # Process frame
            search_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Each method for template matching
            methods = [cv2.TM_CCOEFF,
                       cv2.TM_CCOEFF_NORMED,
                       cv2.TM_CCORR,
                       cv2.TM_CCORR_NORMED,
                       cv2.TM_SQDIFF,
                       cv2.TM_SQDIFF_NORMED]
            titles = ['cv2.TM_CCOEFF',
                      'cv2.TM_CCOEFF_NORMED',
                      'cv2.TM_CCORR',
                      'cv2.TM_CCORR_NORMED',
                      'cv2.TM_SQDIFF',
                      'cv2.TM_SQDIFF_NORMED']

            for i in range(len(methods)):  # Try each method
                cur_img = search_img.copy()
                template_map = cv2.matchTemplate(search_img, target_img, methods[i])
                _, _, min_loc, max_loc = cv2.minMaxLoc(template_map)

                if methods[i] == cv2.TM_SQDIFF or methods[i] == cv2.TM_SQDIFF_NORMED:  # Save left corner of rect
                    top_left = min_loc  # Special SQDIFF case
                else:
                    top_left = max_loc

                #  Display both pre-process img and postimg with matches boxed using matplotlib
                bottom_right = (top_left[0] + target_width, top_left[1] + target_height)
                cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
                plt.figure()
                plt.subplot(121)
                plt.imshow(template_map)
                plt.title(titles[i])
                plt.subplot(122)
                plt.imshow(cur_img)
            plt.show()

    cap.release()
    cv2.destroyAllWindows()


# Read in video as CV2 object, loop through each frame and send for processing
def video_search(target, target_text, video):
    target_img = cv2.imread(target)
    cap = cv2.VideoCapture(video)

    if not cap.isOpened():
        print("Could not open video file")
        exit()
    #frame_rate = cap.get(cv2.CAP_PROP_FPS)
    #frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    #duration = frame_count / frame_rate
    #cap.set(cv2.CAP_PROP_POS_MSEC, frame_rate)
    #current_frame = 0

    marker_timestamps = ['00:00:00']
    # time to frame: hours * 3600 * frame_rate + minutes * 60 * frame_rate + seconds * frame_rate
    while cap.isOpened():  # While video capture is going
        current_frame = current_frame + 1
        ret, frame = cap.read()  # Capture frame of video
        if not ret:  # If failure to capture frame, freak out
            print("Could not read frame")
        else:  # Process frame
            if process_text(frame, target_text):
                # Save time in hr:mn:sc to array, chop off date with [2:]
                marker_timestamps.append(str(datetime.timedelta(milliseconds=cap.get(cv2.CAP_PROP_POS_MSEC)))[2:])

    cap.release()
    cv2.destroyAllWindows()
    return marker_timestamps


# https://pythongeeks.org/python-opencv-text-detection-and-extraction/
# search_img: CV2 object to be searched. search_text: Text to find in image
# Read cv2 object, convert to binary, search for text. Check text for search term.
def process_text(search_img, search_text):
    pytesseract.pytesseract.tesseract_cmd = r"D:/Pytesseract/tesseract.exe"  # Tesseract Location

    search_img_greyscale = cv2.cvtColor(search_img, cv2.COLOR_BGR2GRAY)  # Convert img to greyscale
    #  Convert img to binary
    _, search_img_binary = cv2.threshold(search_img_greyscale, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    search_img_text = pytesseract.image_to_string(Image.fromarray(search_img_binary), config='--psm 11')
    search_img_text = search_img_text.replace("\n", " ")
    # cv2.imshow(search_img_text, search_img)  # DEBUG: Display search image, and search text (As win_title)
    if search_text in search_img_text:
        return True
    else:
        return False


# Take time markers and cut video at the marker - the prefix amount and + the suffix
def cut_video(markers, video, prefix, suffix):
    video_object = mpy.VideoFileClip(video)
    clips = [0]
    for i in range(len(markers)):
        clips[i] = video_object.subclip((markers[i]-prefix), (markers[i]+suffix))
    final_video = mpy.concatenate_videoclips(clips)


def menu():
    selection_size = input("Input 1. Single video or 2. Folder of videos")
    if selection_size == "1":
        video = input("Input video file exact path")
    else:
        folder = input("Input video folder exact path")
    text = input("Input target text")


def main():
    main()


if __name__ == '__main__':
    user_target_img = "C:/Users/Cortes/PycharmProjects/AutoClip/AutoClipTestImg.png"
    user_video = "C:/Users/Cortes/PycharmProjects/AutoClip/AutoClipTestFootage.mp4"
    user_target_text = "Kill +100"
    marker_timestamps = video_search(user_target_img, user_target_text, user_video)
    cut_video(marker_timestamps, user_video, 1, 1)




