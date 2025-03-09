import utils
import easyocr  # OCR
import moviepy.editor as mpy
import cv2
import matplotlib.pyplot as plt


def main():


    if marker_text:
        time_stamps = process_video_ocr(video, marker_coordinates, marker_text, frame_interval, marker_interval, debug) # OCR
    else:
        time_stamps = process_video_tm() # Template Matching
    frames_to_time(time_stamps, fps)
    cut_video(video, time_stamps, prefix, suffix, multi)

def process_video_ocr(video, marker_coordinates, marker_text, frame_interval, marker_interval, debug):
    """
    Read each frame of the video, crop to marker zone, read using Easy OCR, parse for marker, save timestamp

    Loop through total frame count, if frame is mod interval: read and crop frame to marker zone. Preprocess frame and
    read using EasyOCR. If found, save frame count to time stamps, and skip forward by marker interval.
    @param video: Video address
    @param marker_coordinates: x, y, height, width of maker zone
    @param marker_text: Marker text to search for
    @param frame_interval: Frame interval to skip
    @param marker_interval: Marker interval to skip after finding a marker
    @param debug: Display each found frame
    @return:
    """
    time_stamps = [] # Time stamps of found markers, saved as frame numbers
    reader = easyocr.Reader(['en']) # Easy OCR reader

    cv2_capture = cv2.VideoCapture(video)
    if not cv2_capture.isOpened(): # Error check for video
        print("Could not open video file")
        exit()

    frame_total = int(cv2_capture.get(cv2.CAP_PROP_FRAME_COUNT)) # Get frame total
    frame_count = 0 # Frame counter
    frames_read = 0 # Frames read
    marker_found = False
    cv2_capture.set(1, frame_count) # Set video position

    # Loop through video, reading each frame at set interval, preprocess then read set search box with OCR
    while frames_read <= frame_total:
        if marker_found: # If marker is found, skip forward marker interval
            frame_count = frame_count + marker_interval # Increase frame count by marker interval
            cv2_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
            marker_found = False
        frame_count += 1
        if frame_count % frame_interval == 0: # Skip frame intervals
            frames_read +=1
            # Grab frame at frame count
            cv2_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
            frame_grabbed, frame = cv2_capture.read()
            if not frame_grabbed:
                print("Frame empty, possible error")
                break
            if debug: # Shows frame counts
                print(f"Frame Count: {frame_count}, {frames_read}")
            frame = frame[int(marker_coordinates[1]):int(marker_coordinates[1] + marker_coordinates[3]),
                    int(marker_coordinates[0]):int(marker_coordinates[0] + marker_coordinates[2])]  # Crop to marker
            frame = preprocess_ocr(frame) # Preprocess frame

            results = reader.readtext(frame) # Read cropped frame with Easy OCR
            for result in results: # Loop through results
                bbox, text, prob = result
                if marker_text.lower() in text.lower(): # If marker in results
                    time_stamps.append(frame_count) # Save frame count to time stamp list
                    marker_found = True

                    if debug: # Display timestamp, text, and probability of found marker
                        print(f"Timestamp found at {frame_count}")
                        print(f'Text: {text}, Probability: {prob}')
                        plt.figure(figsize=(8,5))
                        plot(1, 1, 1, 'Marker Zone', frame)  # Display marker zone
                        plt.tight_layout()
                        plt.show()

    cv2_capture.release()
    cv2.destroyAllWindows()
    return time_stamps

def preprocess_ocr(frame):
    """
    Preprocess cv2 image for OCR
    Greyscale, invert, and binarization
    @param frame: CV2 image
    @return: cv2 image post transformation
    """
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Greyscale
    frame = cv2.bitwise_not(frame) # Invert
    frame = cv2.adaptiveThreshold(frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY, 11, 2) # Binarization
    return frame

# def process_video_tm(video, marker_coordinates, frame_interval, marker_interval, debug):
#     time_stamps = [] # Time stamps of found markers, saved as frame numbers
#     cv2_capture = cv2.VideoCapture(video)
#     if not cv2_capture.isOpened(): # Error check for video
#         print("Could not open video file")
#         exit()
#
#     frame_total = int(cv2_capture.get(cv2.CAP_PROP_FRAME_COUNT)) # Get frame total
#     frame_count = 0 # Frame counter
#     frames_read = 0 # Frames read
#     marker_found = False
#     cv2_capture.set(1, frame_count) # Set video position
#
#     # Loop through video, reading each frame at set interval, preprocess then read set search box with OCR
#     while frames_read <= frame_total:
#         if marker_found: # If marker is found, skip forward marker interval
#             frame_count = frame_count + marker_interval # Increase frame count by marker interval
#             cv2_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
#             marker_found = False
#         frame_count += 1
#         if frame_count % frame_interval == 0: # Skip frame intervals
#             frames_read +=1
#             # Grab frame at frame count
#             cv2_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
#             frame_grabbed, frame = cv2_capture.read()
#             if not frame_grabbed:
#                 print("Frame empty, possible error")
#                 break
#             if debug: # Shows frame counts
#                 print(f"Frame Count: {frame_count}, {frames_read}")
#             frame = frame[int(marker_coordinates[1]):int(marker_coordinates[1] + marker_coordinates[3]),
#                     int(marker_coordinates[0]):int(marker_coordinates[0] + marker_coordinates[2])]  # Crop to marker
#             frame = preprocess_tm(frame) # Preprocess frame
#
#             result = cv2.matchTemplate(frame, marker, cv2.TM_CCOEFF_NORMED) # Template match
#             min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result) # Match Location
#             top_left = max_loc # Match coordinate
#
#             results = reader.readtext(frame) # Read cropped frame with Easy OCR
#             for result in results: # Loop through results
#                 bbox, text, prob = result
#                 if marker_text.lower() in text.lower(): # If marker in results
#                     time_stamps.append(frame_count) # Save frame count to time stamp list
#                     marker_found = True
#
#                     if debug: # Display timestamp, text, and probability of found marker
#                         print(f"Timestamp found at {frame_count}")
#                         print(f'Text: {text}, Probability: {prob}')
#                         plt.figure(figsize=(8,5))
#                         plot(1, 1, 1, 'Marker Zone', frame)  # Display marker zone
#                         plt.tight_layout()
#                         plt.show()
#                     if debug:  # Show original, crop zone, and match
#                         plt.figure(figsize=(12, 8))
#
#                         plot(1, 2, 1, 'Original Image', original)  # Original Image
#
#                         # Highlight search area in the original image
#                         x_start, y_start, w, h = marker_coordinates
#                         cv2.rectangle(original, (x_start, y_start), (x_start + w, y_start + h), (255, 0, 0), 2)
#
#                         # Draw the rectangle around the detected template location
#                         cv2.rectangle(original, top_left,
#                                       (top_left[0] + marker.shape[1], top_left[1] + marker.shape[0]),
#                                       (0, 255, 0), 2)
#
#                         plot(1, 2, 2, 'Matching Result', result)  # Template Matching Result
#                         plt.tight_layout()
#                         plt.show()
#
#     cv2_capture.release()
#     cv2.destroyAllWindows()
#     return time_stamps

# def preprocess_tm(frame, debug):
#     # TODO: Fix all of this
#     """
#     Preprocess frame using cv2 functions for better effect in template matching
#
#     Greyscale, normalization, gaussian blur, histogram equalization, edge detection, binarization
#     @param frame: cv2 image
#     @param debug: Show each transformation
#     @return: frame post transformation
#     """
#     if debug:
#         plt.figure(figsize=(12, 8))
#         plot(4,2,1,'Original Frame', frame)
#     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Greyscale
#     if debug:
#         plot(4,2,2,'Grayscale', frame)
#     frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX) # Normalize
#     if debug:
#         plot(4,2,3,'Normalize', frame)
#     frame = cv2.GaussianBlur(frame, (5, 5), 0) # Blur
#     if debug:
#         plot(4,2,4,'Gaussian Blur', frame)
#     frame = cv2.equalizeHist(frame) # Histogram equalization - contrast
#     if debug:
#         plot(4,2,5,'Histogram equal', frame)
#     frame = cv2.Canny(frame, 50, 150) # Edge Detection using canny - more shape focus
#     if debug:
#         plot(4,2,6,'Edge Detection', frame)
#     _, frame = cv2.threshold(frame, 127, 255, cv2.THRESH_BINARY) # Binarization
#     if debug:
#         plot(4,2,7,'Threshold', frame)
#         plt.tight_layout()
#         plt.show()
#     return frame

def frames_to_time(time_stamps, fps):
    """
    Convert frame counts in timestamp array to seconds
    @param time_stamps: Array of frame counts
    @param fps: Frames per second of video
    """
    for i in range(len(time_stamps)):
        time_stamps[i] = time_stamps[i] / fps

def cut_video(video, markers, prefix, suffix, multi):
    """
    Using marker timestamps, cut into clips using affixes, and concatenate and save or save clips based on multi
    @param video: Video address
    @param markers: Time stamp markers as seconds
    @param prefix: Time to cut before marker as seconds
    @param suffix: Time to cut after marker as seconds
    @param multi: Bool for multi clip save or concatenated clip
    """
    # TODO: Add user title option; Change clip title to timestamp?
    video_object = mpy.VideoFileClip(video)
    clips = []
    for i in range(len(markers)): # Loop through markers and create a subclip starting at marker - & + affixes
        clips.append(video_object.subclip((markers[i]-prefix), (markers[i]+suffix)))
    if multi: # If multi save loop through clips and save all to file
        for i in range(len(clips)):
            clip_title = f"debug{i}" # Clip title and add clip number
            clips[i].write_videofile(clip_title)
    else: # Concatenate all clips into one clip
        final_video = mpy.concatenate_videoclips(clips)
        final_video.write_videofile('autoclip_debug.mp4')
        # Clean up
        final_video.close()
    for i in range(len(clips)):
        clips[i].close()
    video_object.close()

if __name__ == '__main__':
    main()