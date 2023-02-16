import glob
import matplotlib.pyplot as plt
import os
from pathlib import Path

from src.audio.utils.constants import VIDEOS_DIR

# Perform certain communication pattern evaluations on the rttm file
class ComPatternAnalysis:
    def __init__(self, args) -> None:
        
        self.video_name = args.get("VIDEO_NAME","001")
        
        rttm_file_path = self.get_rttm_path(self.video_name)
        
        # Read rttm file
        self.myfile = open(rttm_file_path, "r")
        
        # Initialize turn taking dictionary
        self.turn_taking = {}
        
        start_time = []
        end_time = []
        speaker_id = []

        # store speaker ID, start time and end time in a list
        for line in self.myfile:
            split_line = line.split()

            # creating 3 lists (speaker, start, end)
            # get start and end time in milliseconds
            start_time.append(round(float(split_line[3])*1000))
            end_time.append(round(float(split_line[3])*1000 + float(split_line[4])*1000))
            speaker_id.append(split_line[7])
            
        # find unique speaker IDs
        unique_speaker_id = list(set(speaker_id))
        
        # List including all start and end times of all speakers
        self.speaker_overview = []

        # find indices of unique speaker IDs, get start and end time of each speaker and concatenate all audio segments of the same speaker
        for speaker in unique_speaker_id:
            # find indices of speaker
            indices = [i for i, x in enumerate(speaker_id) if x == speaker]

            # get start and end time of speaker
            start_time_speaker = [start_time[i] for i in indices]
            end_time_speaker = [end_time[i] for i in indices]
            
            # Make sure that the start and end times are sorted
            start_time_speaker.sort()
            end_time_speaker.sort()
            
            self.speaker_overview.append([speaker, start_time_speaker, end_time_speaker])
            
    def get_rttm_path(self, video_name) -> str:
        
        rttm_file_path = str(VIDEOS_DIR / video_name / "pyavi" / (video_name + ".rttm"))
        
        rttm_path = glob.glob(rttm_file_path)
        
        if not rttm_path:
            raise Exception("No rttm file found for path: " + rttm_file_path)
        else:
            rttm_path = rttm_path[0]

        return rttm_path
            
    def calculate_number_turns(self) -> None:
        """ Calculates the number of turns (number of times each speakers starts a speaking turn) and saves it in a dict 
        """
        
        self.turn_taking["speaker"] = []
        self.turn_taking["turn_taking"] = []
        
        for speaker in self.speaker_overview:
            self.turn_taking["speaker"].append(speaker[0])
            self.turn_taking["turn_taking"].append(len(speaker[1]))
            
        print(self.turn_taking)
        
    def calculate_speaking_duration(self) -> None:
        """ Calculates the speaking duration of each speaker and saves it in a list 
        """
        
        self.speaking_duration = []
        
        for speaker in self.speaker_overview:
            self.speaking_duration.append(sum(speaker[2]) - sum(speaker[1]))
            
        print(self.speaking_duration)
        
        
    def visualize_pattern(self, mode) -> None:
        """ Visualizes the communication pattern of the speakers

        Args:
            mode (string): A string indicating the communication pattern to be visualized
        """
        
        if mode == "number_turns":
            plt.bar(self.turn_taking["speaker"], self.turn_taking["turn_taking"])
            plt.xlabel('Speakers')
            plt.ylabel('Number of turns')
            plt.title('Bar Chart')
            plt.show()
            #plt.waitforbuttonpress()
            