import easyocr  # OCR
import moviepy.editor as mpy
import cv2
import matplotlib.pyplot as plt
from multiprocessing import Pool

# TODO: Plan to fix:
#

def main():
    # TODO: Add template matching
    debug = True  # Show all frames being processed
    if debug:
        video = "C:/Users/Cortes/PycharmProjects/AutoClip/Destiny Iron Banner Destruction.mp4"
        fps = get_fps(video)
        prefix, suffix = 5, 5
        marker_coordinates = (511, 306, 70, 103)
        marker_text = "ki"
        frame_interval = 20  # Interval to skip frames
        marker_interval = 90  # Duration of marker
        multi = False
        gpu = get_gpu_selection()
        time_stamps = process_video(debug, marker_coordinates, marker_text, frame_interval, marker_interval, debug)
    else:
        video = get_video() # Get video address
        fps = get_fps(video) # Get FPS
        prefix, suffix = get_affixes() # Get times to save before and after markers
        frame_interval = get_frame_interval()
        marker_interval = get_marker_interval()
        multi = get_multi()
        marker_coordinates, marker_timestamp = get_marker_zone_location(video)  # Get marker zone location and timestamp
        marker = get_marker(marker_timestamp, video)  # Get marker image
        marker_text = get_marker_text() # Ask for marker text
        gpu = get_gpu_selection() # Get GPU selection for CUDA
        if marker_text: # This was changed, to just be a bool passed to process video rather than 2 separate functions
            time_stamps = process_video_ocr(video, marker_coordinates, marker_text, frame_interval, marker_interval, debug)
        else:
            time_stamps = process_video_tm() # Template Matching
    frames_to_time(time_stamps, fps)
    cut_video(video, time_stamps, prefix, suffix, multi)



def get_video():
    """
    Get video address from user
    @return: video address
    """
    video = input("Input video address: ")
    return video.replace('\\', '/') # Convert backslashes

def get_fps(video):
    """
    Get fps from video file by opening as cv2 object and getting fps
    @param video: video address
    @return: fps
    """
    cv2_capture = cv2.VideoCapture(video)
    if not cv2_capture.isOpened():
        print("Could not open video file")
        exit()
    fps = cv2_capture.get(cv2.CAP_PROP_FPS)
    cv2_capture.release()
    cv2.destroyAllWindows()
    return fps

def get_affixes():
    """
    Get amount of time to cut before and after every marker in timestamp form
    Convert timestamp in HH:MM:SS format to seconds
    @return: prefix and suffix in seconds
    """
    prefix = input("Input time to cut before markers(e.g. 01:00): ")
    prefix = timestamp_to_seconds(prefix) # Convert timestamp to seconds
    suffix = input("Input time to cut after markers(e.g. 01:00): ")
    suffix = timestamp_to_seconds(suffix)
    return prefix, suffix

def get_frame_interval():
    """
    Get frame interval from user
    @return: frame interval
    """
    return input("Input how many frames to skip (30 @ 30fps, would mean checking a frame every second)")

def get_marker_interval():
    """
    Get marker interval from user as HH:MM:SS and convert to seconds
    @return: marker interval in seconds
    """
    marker_interval =  input("Input marker interval HH:MM:SS (Length of time of marker - "
                             "Destiny 1 kill marker lasts on screen for 3 seconds): ")
    return timestamp_to_seconds(marker_interval)

def get_multi():
    """
    Get if user wants multi clips or single compilation
    @return: Multi bool
    """
    multi = input("Type Y to save found markers as multiple clips. N for a single compilation of all clips")
    if multi.lower() == 'y':
        return True
    else:
        return False

def timestamp_to_seconds(timestamp):
    """
    Convert timestamp in HH:MM:SS format to seconds
    @param timestamp:
    @return: time in seconds
    """
    seconds = 0
    for block in timestamp.split(':'):
        seconds = seconds * 60 + int(block, 10)
    return seconds

def get_marker_zone_location(video):
    """
    Get marker zone coordinates from user by using rectangle selector

    Ask user for timestamp in HH:MM:SS format and convert to seconds
    Open frame at time stamp given
    User selects marker using rectangle selector (Select ROI cv2 function)
    @param video: Video address
    @return: marker coordinates in x, y, height, width
             marker timestamp in seconds
    """
    marker_timestamp = input("Input timestamp(HH:MM:SS) where search marker will show: ")
    marker_timestamp = timestamp_to_seconds(marker_timestamp) # Convert timestamp to seconds

    # Open video and set it to frame given by user
    cv2_capture = cv2.VideoCapture(video)
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
    marker_coordinates = marker_rect[0], marker_rect[1], marker_rect[2], marker_rect[3] # Store zone coordinates
    cv2_capture.release()
    cv2.destroyAllWindows()
    return marker_coordinates, marker_timestamp

def get_marker(marker_timestamp, video):
    """
    Get marker at timestamp, and preprocess
    @param marker_timestamp: Marker location timestamp in seconds
    @param video: Video address
    @return: Marker as a cv2 image, preprocessed
    """
    # Open video and set to timestamp
    cv2_capture = cv2.VideoCapture(video)
    if not cv2_capture.isOpened():  # Error check for video
        print("Could not open video file")
        exit()
    cv2_capture.set(1, marker_timestamp)
    frame_grabbed, frame = cv2_capture.read()
    if not frame_grabbed:
        print("Frame empty, possible error")
        exit()

    marker_rect = cv2.selectROI("Select marker", frame, False)  # Select ROI
    marker = preprocess_tm(marker_rect, True) # Preprocess marker
    cv2_capture.release()
    cv2.destroyAllWindows()
    return marker

def plot(row, col, index, text, frame):
    """
    Creates a subplot using matplotlib
    @param row: row count
    @param col: column count
    @param index: index
    @param text: description
    @param frame: image
    """
    plt.subplot(row, col, index)
    plt.title(text)
    plt.imshow(frame, cmap='gray')
    plt.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)

def get_marker_text():
    return input("Input search text for OCR, leave blank for template matching")

def get_gpu_selection():
    """
    Return CUDA compatibility
    @return: Bool for CUDA compatibility
    """
    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
        return True
    else:
        return False

# def process_video_ocr(video, marker_coordinates, marker_text, frame_interval, marker_interval, debug):
#     """
#     Read each frame of the video, crop to marker zone, read using Easy OCR, parse for marker, save timestamp
#
#     Loop through total frame count, if frame is mod interval: read and crop frame to marker zone. Preprocess frame and
#     read using EasyOCR. If found, save frame count to time stamps, and skip forward by marker interval.
#     @param video: Video address
#     @param marker_coordinates: x, y, height, width of maker zone
#     @param marker_text: Marker text to search for
#     @param frame_interval: Frame interval to skip
#     @param marker_interval: Marker interval to skip after finding a marker
#     @param debug: Display each found frame
#     @return:
#     """
#     time_stamps = [] # Time stamps of found markers, saved as frame numbers
#     reader = easyocr.Reader(['en']) # Easy OCR reader
#
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
#             frame = preprocess_ocr(frame) # Preprocess frame
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
#
#     cv2_capture.release()
#     cv2.destroyAllWindows()
#     return time_stamps

def process_video(video, tm, ocr, marker, marker_coordinates, marker_text, frame_interval, marker_interval, gpu, debug):
    """
    Read each frame of the video, crop to marker zone, read using Easy OCR, parse for marker, save timestamp

    Loop through total frame count, if frame is mod interval: read and crop frame to marker zone. Preprocess frame and
    read using EasyOCR. If found, save frame count to time stamps, and skip forward by marker interval.
    @param tm: Template matching bool
    @param ocr: Optical character recognition bool
    @param video: Video address
    @param marker_coordinates: x, y, height, width of maker zone
    @param marker_text: Marker text to search for
    @param frame_interval: Frame interval to skip
    @param marker_interval: Marker interval to skip after finding a marker
    @param gpu: Bool for CUDA support
    @param debug: Display each found frame
    @return: Timestamps as frame counts
    """
    time_stamps = []  # Time stamps of found markers, saved as frame numbers
    if ocr:
        reader = easyocr.Reader(['en'])  # Easy OCR reader

    cv2_capture = cv2.VideoCapture(video)
    if not cv2_capture.isOpened():  # Error check for video
        print("Could not open video file")
        exit()

    frame_total = int(cv2_capture.get(cv2.CAP_PROP_FRAME_COUNT))  # Get frame total
    frame_count = 0  # Frame counter
    frames_read = 0  # Frames read
    marker_found = False
    cv2_capture.set(1, frame_count)  # Set video position

    # Loop through video, reading each frame at set interval, preprocess then read set search box with OCR
    while frames_read <= frame_total:
        if marker_found:  # If marker is found, skip forward marker interval
            frame_count = frame_count + marker_interval  # Increase frame count by marker interval
            cv2_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
            marker_found = False
        frame_count += 1
        if frame_count % frame_interval == 0:  # Skip frame intervals
            frames_read += 1
            # Grab frame at frame count
            cv2_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
            frame_grabbed, frame = cv2_capture.read()
            if not frame_grabbed:
                print("Frame empty, possible error")
                break
            if debug:  # Shows frame counts
                print(f"Frame Count: {frame_count}, {frames_read}")
            frame = frame[int(marker_coordinates[1]):int(marker_coordinates[1] + marker_coordinates[3]),
                    int(marker_coordinates[0]):int(marker_coordinates[0] + marker_coordinates[2])]  # Crop to marker
            if tm:
                frame = preprocess_tm(frame) # Preprocess frame
                args = [frame, marker]
                with Pool() as pool:
                    if gpu:
                        results = pool.map(template_matching_cuda, args)
                    else:
                        results = pool.map(template_matching(), args)
            elif ocr:
                frame = preprocess_ocr(frame) # Preprocess frame
                results = reader.readtext(frame)  # Read cropped frame with Easy OCR
                for result in results:  # Loop through results
                    bbox, text, prob = result
                    if marker_text.lower() in text.lower():  # If marker in results
                        time_stamps.append(frame_count)  # Save frame count to time stamp list
                        marker_found = True

                        if debug:  # Display timestamp, text, and probability of found marker
                            print(f"Timestamp found at {frame_count}")
                            print(f'Text: {text}, Probability: {prob}')
                            plt.figure(figsize=(8, 5))
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

def preprocess_tm(frame, debug):
    # TODO: Fix all of this
    """
    Preprocess frame using cv2 functions for better effect in template matching

    Greyscale, normalization, gaussian blur, histogram equalization, edge detection, binarization
    @param frame: cv2 image
    @param debug: Show each transformation
    @return: frame post transformation
    """
    if debug:
        plt.figure(figsize=(12, 8))
        plot(4,2,1,'Original Frame', frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Greyscale
    if debug:
        plot(4,2,2,'Grayscale', frame)
    frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX) # Normalize
    if debug:
        plot(4,2,3,'Normalize', frame)
    frame = cv2.GaussianBlur(frame, (5, 5), 0) # Blur
    if debug:
        plot(4,2,4,'Gaussian Blur', frame)
    frame = cv2.equalizeHist(frame) # Histogram equalization - contrast
    if debug:
        plot(4,2,5,'Histogram equal', frame)
    frame = cv2.Canny(frame, 50, 150) # Edge Detection using canny - more shape focus
    if debug:
        plot(4,2,6,'Edge Detection', frame)
    _, frame = cv2.threshold(frame, 127, 255, cv2.THRESH_BINARY) # Binarization
    if debug:
        plot(4,2,7,'Threshold', frame)
        plt.tight_layout()
        plt.show()
    return frame

def template_matching_cuda(args):
    frame, marker = args

    # Upload images to the GPU
    gpu_frame = cv2.cuda_GpuMat()
    gpu_frame.upload(frame)
    gpu_marker = cv2.cuda_GpuMat()
    gpu_marker.upload(marker)


    result = cv2.cuda.templateMatching(gpu_frame, gpu_marker, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv2.minMaxLoc(result) # Get matching amount

    result_cpu = result.download()
    return result_cpu, max_val

def template_matching(args):
    frame, marker = args
    result = cv2.matchTemplate(frame, marker, cv2.TM_CCOEFF_NORMED)

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