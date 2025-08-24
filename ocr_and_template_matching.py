import easyocr
import cv2
import debug_plotting
import utils


def process_video(config):
    time_stamps = []
    cv2_capture = cv2.VideoCapture(config['video'])
    if not cv2_capture.isOpened():
        print("Could not open video file")
        exit()

    frame_total = int(cv2_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count, frames_read = 0, 0
    marker_found = False
    cv2_capture.set(1, frame_count)

    if config['marker_text']:
        reader = easyocr.Reader(['en'])

    while frames_read <= frame_total:
        if marker_found:
            frame_count += config['marker_interval']
            cv2_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
            marker_found = False
        frame_count += 1
        if frame_count % config['frame_interval'] == 0:
            frames_read += 1
            cv2_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
            frame_grabbed, frame = cv2_capture.read()
            if not frame_grabbed:
                print("Frame empty, possible error")
                break
            coords = config['marker_coordinates']
            frame_crop = frame[int(coords[1]):int(coords[1] + coords[3]),
                               int(coords[0]):int(coords[0] + coords[2])]
            if config['marker_text']:
                frame_proc = utils.preprocess_ocr(frame_crop)
                results = reader.readtext(frame_proc)
                for bbox, text, prob in results:
                    if config['marker_text'].lower() in text.lower():
                        time_stamps.append(frame_count)
                        marker_found = True
                        if config['debug']:
                            print(f"Timestamp found at {frame_count}, Text: {text}, Probability: {prob}")
                            debug_plotting.plot_marker_zone(frame_crop)
            else:
                frame_proc = utils.preprocess_tm(frame_crop, config['debug'])
                result = cv2.matchTemplate(frame_proc, config['marker'], cv2.TM_CCOEFF_NORMED)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
                threshold = config.get('tm_threshold', 0.8)
                if max_val >= threshold:
                    time_stamps.append(frame_count)
                    marker_found = True
                    if config.get('debug', False):
                        print(f"TM match at {frame_count}, score: {max_val}")
                        debug_plotting.plot_tm_result(frame,config['marker_coordinates'],
                                                      config['marker'], max_loc, result)

    cv2_capture.release()
    return time_stamps
