import os
import config_handler
import cv2


def get_video_params():
    """Gets video params from config or from user input.

    Returns:
        dict: A dictionary containing video parameters with keys:
            - "video" (str): The video file path.
            - "fps" (int): Frames per second.
            - "prefix" (int): Time to save before marker.
            - "suffix" (int): Time to save after marker.
            - "frame_interval" (int): Interval to skip frames.
            - "marker_interval" (int): Duration of the marker.
            - "marker_zone_coordinates" (tuple[int, int, int, int]): The marker zone bounding box (x, y, width, height).
            - "marker_coordinates" = (tuple[int, int, int, int]): The marker zone bounding box (x, y, width, height).
            - "marker_timestamp" (int): Marker timestamp in seconds.
            - "marker_text" (str): The text used for marker detection.
            - "multi" (bool): Whether to save clips as multiple files or a single compilation.
            - "debug" (bool): Whether to enable debug mode.
    """
    config = config_handler.load_config()

    # Ask the user if they want to use existing config
    use_config = False
    if config:
        use_config = input("Use saved configuration? [Y/n]: ").strip().lower() == "y"

    if use_config:
        print("Using saved configuration.")
        return config  # Load existing config
    else:
        video = get_video()  # Get video address
        fps = get_fps(video)  # Get FPS
        frame_interval = get_frame_interval()
        marker_interval = get_marker_interval(fps)
        marker_zone_coordinates, marker_timestamp = get_marker_zone_location(video)  # Get marker zone location and timestamp
        marker_coordinates = get_marker(video, marker_timestamp, marker_zone_coordinates)  # Get marker coordinates
        marker_text = get_marker_text()  # Ask for marker text
        prefix, suffix = get_affixes()  # Get times to save before and after markers
        multi = get_multi()
        debug = get_mode()  # Get debug mode


        config = {
            "video": video,
            "fps": fps,
            "frame_interval": frame_interval,
            "marker_interval": marker_interval,
            "marker_zone_coordinates": marker_zone_coordinates,
            "marker_timestamp": marker_timestamp,
            "marker_coordinates": marker_coordinates,
            "marker_text": marker_text,
            "prefix": prefix,
            "suffix": suffix,
            "multi": multi,
            "debug": debug
        }
        save_choice = input("Save settings? [Y/n]: ").strip().lower()
        if save_choice == "y":
            config_handler.save_config(config)
            print("Settings saved!")

        return config


def get_video():
    """
    Get and validate video address from user, handling quoted paths and normalizing separators.
    Reprompts until a valid video file is provided.

    Returns:
        str: Normalized video file path
    """
    while True:
        # Get input and strip quotes and whitespace
        video = input("Input video address: ").strip().strip('"\'')

        # Normalize path separators and make absolute
        video = os.path.abspath(video.replace('\\', '/'))

        # Check if file exists
        if not os.path.isfile(video):
            print(f"File not found: {video}")
            continue

        # Try opening with OpenCV to validate it's a video file
        cap = cv2.VideoCapture(video)
        if not cap.isOpened():
            print(f"Could not open as video file: {video}")
            cap.release()
            continue

        cap.release()
        return video

def get_fps(video):
    """
    Get fps from video file by opening as cv2 object and getting fps
    @param video: video address
    @return: fps
    """
    cv2_capture = cv2.VideoCapture(video)
    if not cv2_capture.isOpened():
        print("Could not open video file - get_fps")
        exit()
    fps = cv2_capture.get(cv2.CAP_PROP_FPS)
    cv2_capture.release()
    return fps

def get_affixes():
    """
    Get amount of time to cut before and after every marker in timestamp form
    Convert timestamp in HH:MM:SS format to seconds
    @return: prefix and suffix in seconds
    """
    prefix = parse_time_input("Input prefix (time to cut BEFORE markers) (HH:MM:SS or MM:SS or S): ")
    suffix = parse_time_input("Input suffix (time to cut AFTER markers) (HH:MM:SS or MM:SS or S): ")
    return prefix, suffix

def get_frame_interval():
    """
    Get frame interval from user and ensures it's an integer.
    @return: frame interval as an integer
    """
    while True:
        try:
            interval_str = input("Input frame skip interval (e.g., 30 for 30fps to check every second): ")
            return int(interval_str)
        except ValueError:
            print("Invalid input. Please enter a whole number.")

def get_marker_interval(fps):
    """
    Get marker interval from user and convert to frames

    Args:
        fps (float): Video frames per second

    Returns:
        int: Number of frames to skip after finding a marker
    """
    seconds = parse_time_input("Input marker skip interval (Length of time to skip after a marker is found) (HH:MM:SS or MM:SS or S): ")
    frames = int(seconds * fps)
    print(f"    Converting {seconds} seconds to {frames} frames at {fps} FPS")
    return frames


def get_multi():
    """
    Get if user wants multi clips or single compilation
    @return: Multi bool
    """
    multi = input("Input Multi Mode Selection (Type Y to save found markers as multiple clips. N for a single compilation of all clips): ")
    if multi.lower() == 'y':
        return True
    else:
        return False


def get_marker_zone_location(video):
    """
    Get marker zone coordinates from user by using rectangle selector.

    Asks user for a timestamp, shows the frame, and asks for confirmation.
    If not confirmed, it re-prompts for the timestamp.
    Once confirmed, the user selects the marker area using a rectangle selector.

    Args:
        video (str): The path to the video file.

    Returns:
        tuple: A tuple containing:
            - tuple[int, int, int, int]: Marker coordinates (x, y, width, height).
            - int: The confirmed marker timestamp in seconds.
    """
    cv2_capture = cv2.VideoCapture(video)
    if not cv2_capture.isOpened():
        print("Could not open video file")
        exit()

    marker_fps = cv2_capture.get(cv2.CAP_PROP_FPS)
    frame = None
    marker_timestamp_seconds = 0

    while True:
        marker_timestamp_seconds = parse_time_input("Input sample marker timestamp (HH:MM:SS or MM:SS or S): ")
        frame_number = int(marker_timestamp_seconds * marker_fps)
        cv2_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

        frame_grabbed, frame = cv2_capture.read()
        if not frame_grabbed:
            print("Frame empty, possible error. Please try another timestamp.")
            continue

        cv2.imshow("Confirm Frame", frame)
        cv2.waitKey(1)
        print("Showing frame at the specified timestamp. Check the window.")

        # Wait for user input in the console, not the OpenCV window
        confirm = input("Is this the correct frame? [Y/n]: ").strip().lower()

        cv2.destroyWindow("Confirm Frame")

        if confirm == 'y' or confirm == '':
            break

    # Proceed with ROI selection on the confirmed frame
    marker_rect = cv2.selectROI("Select marker location", frame, False)
    cv2.destroyAllWindows()

    marker_coordinates = (marker_rect[0], marker_rect[1], marker_rect[2], marker_rect[3])
    cv2_capture.release()

    return marker_coordinates, marker_timestamp_seconds

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


def preprocess_tm(frame):
    """
    Preprocess frame using cv2 functions for better effect in template matching

    Greyscale, normalization, gaussian blur, histogram equalization, edge detection, binarization
    @param frame: cv2 image
    @return: frame post transformation
    """
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Greyscale
    frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX) # Normalize
    frame = cv2.GaussianBlur(frame, (5, 5), 0) # Blur
    frame = cv2.equalizeHist(frame) # Histogram equalization - contrast
    frame = cv2.Canny(frame, 50, 150) # Edge Detection using canny - more shape focus
    _, frame = cv2.threshold(frame, 127, 255, cv2.THRESH_BINARY) # Binarization
    return frame


def get_marker(video, marker_timestamp, marker_zone_coordinates):
        """
        From a sample frame, crops to the marker zone, then allows the user
        to select the specific marker within that zone.
        @param video: Video address.
        @param marker_timestamp: Marker location timestamp in seconds.
        @param marker_zone_coordinates: A tuple (x, y, w, h) for the search zone ROI.
        @return: A tuple (x, y, w, h) for the specific marker, relative to the full frame.
        """
        # Open video and set to timestamp
        cv2_capture = cv2.VideoCapture(video)
        if not cv2_capture.isOpened():
            print("Could not open video file")
            exit()

        fps = cv2_capture.get(cv2.CAP_PROP_FPS)
        frame_number = int(marker_timestamp * fps)
        cv2_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

        frame_grabbed, frame = cv2_capture.read()
        if not frame_grabbed:
            print("Frame empty, possible error")
            exit()
        cv2_capture.release()

        # Crop the frame to the marker zone for easier selection
        zx, zy, zw, zh = marker_zone_coordinates
        zone_image = frame[zy:zy+zh, zx:zx+zw]

        # Allow user to select the marker within the zone
        marker_rect_relative = cv2.selectROI("Select the marker within this zone", zone_image, False)
        cv2.destroyAllWindows()

        # Convert relative marker coordinates to absolute coordinates on the full frame
        rx, ry, rw, rh = marker_rect_relative
        absolute_x = zx + rx
        absolute_y = zy + ry
        marker_coordinates_absolute = (absolute_x, absolute_y, rw, rh)

        return marker_coordinates_absolute


def get_marker_text():
    return input("Input search text for OCR (Leave blank for Template Matching only): ")


def frames_to_time(time_stamps, fps):
    """
    Convert frame counts in timestamp array to seconds
    @param time_stamps: Array of frame counts
    @param fps: Frames per second of video
    """
    for i in range(len(time_stamps)):
        time_stamps[i] = time_stamps[i] / fps

def get_mode():
    mode = input("Type 1 for Debug mode (Displays every frame where markers are found): ")
    if mode == '1':
        return True
    else:
        return False


def parse_time_input(prompt):
    """
    Get and validate time input from user in HH:MM:SS, MM:SS, or seconds format.
    Seconds can be a floating point number.

    Args:
        prompt (str): Input prompt for the user

    Returns:
        float: Time in seconds
    """
    while True:
        time_str = input(prompt).strip()

        if not time_str:
            print("Input cannot be empty.")
            continue

        # Try to parse as a float first (for seconds like 33.5)
        try:
            seconds = float(time_str)
            int_seconds = int(seconds)
            formatted_time = f"{int_seconds//3600:02d}:{(int_seconds%3600)//60:02d}:{int_seconds%60:02d}"
            print(f"    Input: {seconds} seconds (approximately {formatted_time})")
            return seconds
        except ValueError:
            # If not a float, it might be HH:MM:SS or MM:SS
            pass

        try:
            parts = time_str.split(':')
            total_seconds = 0.0

            if len(parts) == 2:  # MM:SS format
                minutes = int(parts[0])
                seconds = float(parts[1])  # Allow float in seconds part
                if 0 <= minutes < 60 and 0 <= seconds < 60:
                    total_seconds = float(minutes * 60) + seconds
                    return total_seconds

            elif len(parts) == 3:  # HH:MM:SS format
                hours = int(parts[0])
                minutes = int(parts[1])
                seconds = float(parts[2])  # Allow float in seconds part
                if 0 <= hours < 24 and 0 <= minutes < 60 and 0 <= seconds < 60:
                    total_seconds = float(hours * 3600 + minutes * 60) + seconds
                    return total_seconds

            print("Invalid format. Use seconds (e.g., 120 or 33.5) or HH:MM:SS (e.g., 01:30 or 00:02:00.5)")
        except ValueError:
            print("Invalid input. Use numbers, decimals, or time format.")