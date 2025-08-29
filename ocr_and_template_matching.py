import easyocr
import cv2
import debug_plotting
import utils
from tqdm import tqdm


def process_video(config):
    time_stamps = []
    cv2_capture = cv2.VideoCapture(config['video'])
    if not cv2_capture.isOpened():
        print("Could not open video file")
        exit()

    # Get marker template if not doing OCR
    marker_template = None
    if not config['marker_text']:
        fps = cv2_capture.get(cv2.CAP_PROP_FPS)
        marker_frame_num = int(config['marker_timestamp'] * fps)
        cv2_capture.set(cv2.CAP_PROP_POS_FRAMES, marker_frame_num)
        ret, frame = cv2_capture.read()
        if not ret:
            print("Could not read frame for marker template.")
            cv2_capture.release()
            exit()
        coords = config['marker_coordinates']
        marker_crop = frame[int(coords[1]):int(coords[1] + coords[3]),
                          int(coords[0]):int(coords[0] + coords[2])]
        marker_template = utils.preprocess_tm(marker_crop)
        cv2_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

    frame_total = int(cv2_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = 0
    marker_found = False
    markers_detected_count = 0
    if config['marker_text']:
        reader = easyocr.Reader(['en'])

    with tqdm(total=frame_total, unit='frame', desc=f"Processing Video | Markers found: {markers_detected_count}") as pbar:
        while frame_count < frame_total:
            pbar.update(frame_count - pbar.n)
            cv2_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
            frame_grabbed, frame = cv2_capture.read()
            if not frame_grabbed:
                print("Frame empty, possible error")
                break

            coords = config['marker_zone_coordinates']
            frame_crop = frame[int(coords[1]):int(coords[1] + coords[3]),
                             int(coords[0]):int(coords[0] + coords[2])]

            if config['marker_text']:  # OCR mode
                frame_proc = utils.preprocess_ocr(frame_crop)
                results = reader.readtext(frame_proc)

                if config.get('debug', False):
                    ocr_text_combined = ' '.join([text for _, text, _ in results])
                    debug_plotting.save_debug_frame(frame_crop, frame_proc, frame_count,
                                                    ocr_text_combined, config['video'])

                for bbox, text, prob in results:
                    if config['marker_text'].lower() in text.lower():
                        time_stamps.append(frame_count)
                        marker_found = True
                        markers_detected_count += 1
                        pbar.set_description(f"Processing Video | Markers found: {markers_detected_count}")
                        if config['debug']:
                            print(f"Timestamp found at {frame_count}, Text: {text}, Probability: {prob}")
                        break
            else:  # Template Matching mode
                frame_proc = utils.preprocess_tm(frame_crop)
                h, w = marker_template.shape
                H, W = frame_proc.shape
                window_step = 10
                max_val = 0
                max_loc = None
                window_locations = []  # Track all window locations

                if h * w * 4 < H * W:  # Use sliding window
                    for y in range(0, H - h + 1, window_step):
                        for x in range(0, W - w + 1, window_step):
                            window = frame_proc[y:y + h, x:x + w]
                            if window.shape == marker_template.shape:
                                window_locations.append((x, y))  # Store window location
                                result = cv2.matchTemplate(window, marker_template, cv2.TM_CCOEFF_NORMED)
                                min_val, curr_max_val, min_loc, curr_max_loc = cv2.minMaxLoc(result)
                                if curr_max_val > max_val:
                                    max_val = curr_max_val
                                    max_loc = (x, y)
                else:
                    result = cv2.matchTemplate(frame_proc, marker_template, cv2.TM_CCOEFF_NORMED)
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
                    window_locations = [(max_loc[0], max_loc[1])]  # Single window for full template match

                threshold = config.get('tm_threshold', 0.8)
                if max_val >= threshold:
                    time_stamps.append(frame_count)
                    marker_found = True
                    markers_detected_count += 1
                    pbar.set_description(f"Processing Video | Markers found: {markers_detected_count}")
                    if config.get('debug', False):
                        print(f"TM match at {frame_count}, score: {max_val}")
                        debug_plotting.save_tm_debug_frame(frame, frame_proc, marker_template,
                                                           frame_count, window_locations, max_loc,
                                                           max_val, config['marker_coordinates'],
                                                           config['video'])

            if marker_found:
                frame_count += config['marker_interval']
                marker_found = False
            else:
                frame_count += config['frame_interval']
        pbar.update(frame_total - pbar.n)

    cv2_capture.release()
    if config.get('debug', False) and not config['marker_text']:
        cv2.destroyAllWindows()
    if config.get('debug', False):
        debug_plotting.create_html_report(config['video'])
    if not time_stamps:
        raise ValueError("No markers were found in the video. Please check your configuration.")
    return time_stamps