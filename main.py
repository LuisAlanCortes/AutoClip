import utils
import ocr_and_template_matching as ocr_tm
import video_editing

def main():
    config = utils.get_video_params()  # Get video params from config or user input
    timestamps = ocr_tm.process_video(config)

    utils.frames_to_time(timestamps, config['fps'])
    video_editing.cut_video(config['video'], timestamps, config['prefix'],
                         config['suffix'], config.get('multi', False))

if __name__ == '__main__':
    main()