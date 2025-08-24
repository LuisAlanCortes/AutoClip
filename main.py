import utils
import ocr_and_template_matching as ocr_tm
import video_editing

def main():
    config = utils.get_video_params()  # Get video params from config or user input
    ocr_tm.process_video(config)

    utils.frames_to_time(config['time_stamps'], config['fps'])
    video_editing.cut_video(config['video'], config['time_stamps'], config['prefix'],
                         config['suffix'], config.get('multi', False))

if __name__ == '__main__':
    main()