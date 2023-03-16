import os
from main import main

from src.audio.utils.constants import VIDEOS_DIR

def custom_sort(folder_name):
    # Split the folder name into parts
    parts = folder_name.split('_')
    
    # Extract the clip number and frame numbers
    clip_num = int(parts[1])
    start_frame = int(parts[2])
    
    # Return a tuple to define the sorting order
    return (clip_num, start_frame)

if __name__ == '__main__':
    # video_path = '/Users/tobiaszeulner/Desktop/Master_Thesis_MIT/Teamwork-audio-analysis/src/audio/videos/001.mp4'
    # main(video_path)

    # Define the path to the top-level directory
    team = 'team_17'
    team_folder = str(VIDEOS_DIR / team)

    # Loop over the subdirectories (i.e., the day folders)
    for day_folder in sorted(os.listdir(team_folder)):
        day_path = os.path.join(team_folder, day_folder)
        if os.path.isdir(day_path):
            # Loop over the video files within each day folder
            
            video_file_names = [f for f in os.listdir(day_path) if f.endswith('.mp4')]
            sorted_video_file_names = sorted(video_file_names, key=custom_sort)
            
            for video_file in sorted_video_file_names:
                # Construct the full path to the video file
                video_path = os.path.join(day_path, video_file)
                # Process the video file (e.g., extract features, perform analysis, etc.)
                main(video_path)