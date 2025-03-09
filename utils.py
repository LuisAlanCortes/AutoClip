import config_handler

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
            - "marker_coordinates" (tuple[int, int, int, int]): The marker's bounding box (x, y, width, height).
            - "marker_text" (str): The text used for marker detection.
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
        prefix, suffix = get_affixes()  # Get times to save before and after markers
        frame_interval = get_frame_interval()
        marker_interval = get_marker_interval()
        multi = get_multi()
        marker_coordinates, marker_timestamp = get_marker_zone_location(video)  # Get marker zone location and timestamp
        marker = get_marker(marker_timestamp, video)  # Get marker image
        marker_text = get_marker_text()  # Ask for marker text

        config = {
            "video": video,
            "fps": fps,
            "prefix": prefix,
            "suffix": suffix,
            "frame_interval": frame_interval,
            "marker_interval": marker_interval,
            "marker_coordinates": marker_coordinates,
            "marker_text": marker_text,
        }
        save_choice = input("Save these settings for next time? [Y/n]: ").strip().lower()
        if save_choice == "y":
            config_handler.save_config(config)
            print("Settings saved!")

        return config

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