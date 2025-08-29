import moviepy as mpy


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
        clips.append(video_object.subclipped((markers[i]-prefix), (markers[i]+suffix)))
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


