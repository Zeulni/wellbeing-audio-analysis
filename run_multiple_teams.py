import os
from main import main

from src.audio.utils.constants import VIDEOS_DIR

if __name__ == '__main__':
    # video_path = '/Users/tobiaszeulner/Desktop/Master_Thesis_MIT/Teamwork-audio-analysis/src/audio/videos/001.mp4'
    # main(video_path)

    # Define the path to the top-level directory
    team = 'team_13'
    team_folder = str(VIDEOS_DIR / team)

    # Loop over the subdirectories (i.e., the day folders)
    for day_folder in sorted(os.listdir(team_folder)):
        day_path = os.path.join(team_folder, day_folder)
        if os.path.isdir(day_path):
            # Loop over the video files within each day folder
            for video_file in sorted(os.listdir(day_path)):
                # Check that the file is an MP4
                if video_file.endswith('.mp4'):
                    # Construct the full path to the video file
                    video_path = os.path.join(day_path, video_file)
                    # Process the video file (e.g., extract features, perform analysis, etc.)
                    main(video_path)