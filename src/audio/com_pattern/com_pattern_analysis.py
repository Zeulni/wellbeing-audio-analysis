import glob
import math
import matplotlib.pyplot as plt
import statistics
import os

from src.audio.utils.constants import VIDEOS_DIR

# Perform certain communication pattern evaluations on the rttm file
class ComPatternAnalysis:
    def __init__(self, args, length_video) -> None:
        
        self.video_name = args.get("VIDEO_NAME","001")
        
        rttm_file_path = self.get_rttm_path(self.video_name)
        
        # Read rttm file
        self.myfile = open(rttm_file_path, "r")
        
        # Get the length of the video (stored in filename)
        self.length_video = length_video
        
        # Initialize the com pattern dictionaries
        self.number_turns = {}
        self.number_turns_share = {}
        
        self.speaking_duration = {}
        self.speaking_duration_share = {}
        self.speaking_duration_share_std = None
        
        start_time = []
        end_time = []
        speaker_id = []

        # store speaker ID, start time and end time in a list
        for line in self.myfile:
            split_line = line.split()

            # creating 3 lists (speaker, start, end)
            # get start and end time in seconds
            start_time.append(round(float(split_line[3])))
            end_time.append(round(float(split_line[3]) + float(split_line[4])))
            speaker_id.append(split_line[7])
            
        # find unique speaker IDs
        unique_speaker_id = list(set(speaker_id))
        
        # Number of speakers
        self.num_speakers = len(unique_speaker_id)
        
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
            
    def run(self) -> None:
        # PERMA score higher for teams that start more conversations (e.g. shorter ones)
        self.calculate_number_turns()
        
        # PERMA score higher for teams that have a equal distribution of turns?
        self.calculate_number_turns_share()
        self.calculate_number_turns_equality()
        
        # PERMA score higher for teams that speak more? (-> calculate one score that indicates how much they are speaking in percent)
        self.calculate_speaking_duration()
        
        # PERMA score higher for teams that have a equal distribution of speaking time? (-> one score that indicates how equaly distributed they are speaking)
        self.calculate_speaking_duration_share()
        self.calculate_speaking_duration_equality()
        
        #overlaps
        self.calculate_amount_overlaps()
        
        # Visualize the communication patterns
        self.visualize_pattern()
            
    def get_rttm_path(self, video_name) -> str:
        
        rttm_folder_path = str(VIDEOS_DIR / video_name / "pyavi")
        rttm_file = glob.glob(os.path.join(rttm_folder_path, "*.rttm"))
        
        if not rttm_file:
            raise Exception("No rttm file found for path: " + rttm_folder_path)
        else:
            rttm_path = rttm_file[0]

        return rttm_path
            
    def calculate_number_turns(self) -> None:
        # Calculates the number of turns (number of times each speakers starts a speaking turn) and saves it in a dict 
        
        self.number_turns["speaker"] = []
        self.number_turns["number_turns"] = []
        
        for speaker in self.speaker_overview:
            self.number_turns["speaker"].append(speaker[0])
            self.number_turns["number_turns"].append(len(speaker[1]))
            
        print(self.number_turns)
        
    # Calculate based on the number_turns dict the share in number of turns of each speaker
    def calculate_number_turns_share(self) -> None:
        
        # Calculate the total number of turns
        total_number_turns = sum(self.number_turns["number_turns"])
        
        # Calculate the share in number of turns of each speaker
        self.number_turns_share["speaker"] = []
        self.number_turns_share["share_number_turns"] = []
        
        for speaker in self.number_turns["speaker"]:
            self.number_turns_share["speaker"].append(speaker)
            share = (self.number_turns["number_turns"][self.number_turns["speaker"].index(speaker)] / total_number_turns)*100
            share = round(share, 1)
            self.number_turns_share["share_number_turns"].append(share)
            
        print(self.number_turns_share)
        
    # Calculating the equality based on the number of turns  
    def calculate_number_turns_equality(self) -> None:
        
        # TODO: Independent of number of length (as normalized by mean), but also ind. of #speakers?
        mean_time = statistics.mean(self.number_turns["number_turns"])
        stdev_time = statistics.stdev(self.number_turns["number_turns"])
        cv = (stdev_time / mean_time) * 100
        
        # TODO: Formula see Ignacio's thesis
        # Calculating the speaker equality under the assumption, that the max speaker duration is a good proxy for the 
        # expected speaking duration that each member should take in a perfectly equal distribution
        # max_cv = (math.sqrt(self.num_speakers) - 1) / math.sqrt(self.num_speakers)
        # speaker_equality = 1 - (cv / max_cv)

        print("Number turns equality (0 would be perfectly equal): ", cv)        
        
    def calculate_speaking_duration(self) -> None:
        # Calculates the speaking duration of each speaker and saves it in a list 
        
        self.speaking_duration["speaker"] = []
        self.speaking_duration["speaking_duration"] = []

        for speaker in self.speaker_overview:
            self.speaking_duration["speaker"].append(speaker[0])
            self.speaking_duration["speaking_duration"].append(sum(speaker[2]) - sum(speaker[1]))
            
        print(self.speaking_duration)
        
    # Calculates based on the speaking_duration dict the share in speaking time of each speaker    
    def calculate_speaking_duration_share(self) -> None:
        
        # Calculate the total speaking duration
        total_speaking_duration = sum(self.speaking_duration["speaking_duration"])
        
        # Calculate the share in speaking time of each speaker
        self.speaking_duration_share["speaker"] = []
        self.speaking_duration_share["share_speaking_time"] = []
        
        for speaker in self.speaking_duration["speaker"]:
            self.speaking_duration_share["speaker"].append(speaker)
            share = (self.speaking_duration["speaking_duration"][self.speaking_duration["speaker"].index(speaker)] / total_speaking_duration)*100
            share = round(share, 1)
            self.speaking_duration_share["share_speaking_time"].append(share)
            
        print(self.speaking_duration_share)
        
    # Calculating the equality based on the speaking duration   
    def calculate_speaking_duration_equality(self) -> None:
        
        # TODO: Independent of number of length (as normalized by mean), but also ind. of #speakers?
        mean_time = statistics.mean(self.speaking_duration["speaking_duration"])
        stdev_time = statistics.stdev(self.speaking_duration["speaking_duration"])
        cv = (stdev_time / mean_time) * 100
        
        # TODO: Formula see Ignacio's thesis
        # Calculating the speaker equality under the assumption, that the max speaker duration is a good proxy for the 
        # expected speaking duration that each member should take in a perfectly equal distribution
        # max_cv = (math.sqrt(self.num_speakers) - 1) / math.sqrt(self.num_speakers)
        # speaker_equality = 1 - (cv / max_cv)

        print("Speaking duration equality (0 would be perfectly equal): ", cv)
        
        
    def calculate_amount_overlaps(self) -> None:

        num_overlaps = 0

        # Iterate through the speaker list and compare speech segments
        for i in range(len(self.speaker_overview)):
            for j in range(i+1, len(self.speaker_overview)):
                # Iterate through all speech segments for speaker i
                for seg_i in range(len(self.speaker_overview[i][1])):
                    # Iterate through all speech segments for speaker j
                    for seg_j in range(len(self.speaker_overview[j][1])):
                        # Check if the speech segments overlap
                        if max(self.speaker_overview[i][1][seg_i], self.speaker_overview[j][1][seg_j]) < min(self.speaker_overview[i][2][seg_i], self.speaker_overview[j][2][seg_j]):
                            num_overlaps += 1
        
        # Normalize the number of overlaps bei the length of the audio snippet and the number of speakers
        # -> #overlaps per minute per speaker
        norm_num_overlaps = (num_overlaps / (self.length_video * self.num_speakers))*60
        
        # Print the number of overlaps
        print("Number of overlaps (per minute per speaker): ", norm_num_overlaps)
        
        
        
    def visualize_pattern(self, mode = "") -> None:
        
        # Create a figure and two subplots (one for each dictionary)
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        # Plot the first dictionary
        axes[0,0].bar(self.number_turns['speaker'], self.number_turns['number_turns'])
        axes[0,0].set_title('Number of turns per speaker')
        axes[0,0].set_ylabel('Number turns')

        # Plot the second dictionary
        axes[0,1].bar(self.speaking_duration['speaker'], self.speaking_duration['speaking_duration'])
        axes[0,1].set_title('Speaking duration per speaker')
        axes[0,1].set_ylabel('Speaking duration (s)')
        
        axes[1,0].bar(self.number_turns_share['speaker'], self.number_turns_share['share_number_turns'])
        axes[1,0].set_title('Number turns share per speaker')
        axes[1,0].set_ylabel('Number turns share in %')
        
        axes[1,1].bar(self.speaking_duration_share['speaker'], self.speaking_duration_share['share_speaking_time'])
        axes[1,1].set_title('Speaking share per speaker')
        axes[1,1].set_ylabel('Speaking share in %')
        
        plt.show()
        
        # # Showing number_turns and speaking_duration in one bar chart (two bars per speaker)
        # # Each one has his own y-axis
        # plt.bar(self.number_turns["speaker"], self.number_turns["number_turns"])
        # plt.title("Number of turns per speaker")
        # plt.xlabel("Speaker")
        # plt.ylabel("Number of turns")
        # plt.show()

            