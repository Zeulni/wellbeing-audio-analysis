import glob
import matplotlib.pyplot as plt
import statistics
import os

from src.audio.com_pattern.turn_taking import TurnTaking
from src.audio.com_pattern.speaking_duration import SpeakingDuration
from src.audio.com_pattern.overlaps import Overlaps

from src.audio.utils.constants import VIDEOS_DIR

# Perform certain communication pattern evaluations on the rttm file
class ComPatternAnalysis:
    def __init__(self, args, length_video) -> None:
        
        self.video_name = args.get("VIDEO_NAME","001")
        self.unit_of_analysis = args.get("UNIT_OF_ANALYSIS", 300)
        
        rttm_file_path = self.get_rttm_path(self.video_name)
        
        # Read rttm file
        self.myfile = open(rttm_file_path, "r")
        
        # Get the length of the video (stored in filename)
        self.length_video = length_video
        
        # Based on the unit of analysis and the length of the video, create a list with the length of each block
        self.block_length = self.get_block_length()
        
        start_time = []
        end_time = []
        speaker_id = []

        # store speaker ID, start time and end time in a list
        for line in self.myfile:
            split_line = line.split()

            # creating 3 lists (speaker, start, end)
            # get start and end time in seconds
            start_time.append(round(float(split_line[3]),2))
            end_time.append(round(float(split_line[3]) + float(split_line[4]),2))
            speaker_id.append(split_line[7])
            
        # find unique speaker IDs
        unique_speaker_id = list(set(speaker_id))
        
        # Number of speakers
        self.num_speakers = len(unique_speaker_id)
        
        # Initialize the communication pattern classes
        self.turn_taking = TurnTaking()
        self.speaking_duration = SpeakingDuration()
        self.overlaps = Overlaps(self.num_speakers)
        
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
        
        self.splitted_speaker_overview = self.split_speaker_overview()
            
    def get_block_length(self):
        block_length = []
        start_time = 0
        while start_time < self.length_video:
            end_time = min(start_time + self.unit_of_analysis, self.length_video)
            block_length.append(end_time - start_time)
            start_time += self.unit_of_analysis
        return block_length
            
    # Split the speaker overview into blocks (based on unit of analysis)
    def split_speaker_overview(self) -> list:
        
        block_duration = self.unit_of_analysis

        block_speaker_data = []
        start_time = 0
        while start_time < self.length_video:
            end_time = min(start_time + block_duration, self.length_video)
            block_data = []
            for speaker_id, speaker_starts, speaker_ends in self.speaker_overview:
                speaker_block_starts = []
                speaker_block_ends = []
                for i in range(len(speaker_starts)):
                    segment_start = max(start_time, speaker_starts[i])
                    segment_end = min(end_time, speaker_ends[i])
                    if segment_start < segment_end:
                        speaker_block_starts.append(segment_start)
                        speaker_block_ends.append(segment_end)
                    elif segment_start == segment_end and i < len(speaker_starts)-1 and speaker_ends[i] < speaker_starts[i+1]:
                        # Special case when segment is exactly at block boundary and there is a gap before the next segment
                        speaker_block_starts.append(segment_start)
                        speaker_block_ends.append(segment_end)
                        speaker_block_starts.append(segment_end)
                        speaker_block_ends.append(speaker_starts[i+1])
                if speaker_block_starts:
                    block_data.append([speaker_id, speaker_block_starts, speaker_block_ends])
            block_speaker_data.append(block_data)
            start_time += block_duration

            
        return block_speaker_data
   
            
    def run(self) -> None:
        
        # For each unit of analysis (block) perform the following calculations
        for block_id, speaker_overview in enumerate(self.splitted_speaker_overview):
            # PERMA score higher for teams that start more conversations (e.g. shorter ones)
            number_turns = self.turn_taking.calculate_number_turns(speaker_overview)

            # PERMA score higher for teams that have a equal distribution of turns?
            # number_turns_share = self.turn_taking.calculate_number_turns_share(number_turns)

            number_turns_equality = self.turn_taking.calculate_number_turns_equality(number_turns)
            print("Number turns equality (0 would be perfectly equal): ", number_turns_equality)
            self.turn_taking.blocks_number_turns_equality["block"].append(block_id)
            self.turn_taking.blocks_number_turns_equality["number_turns_equality"].append(number_turns_equality)
        
            # PERMA score higher for teams that speak more? (-> calculate one score that indicates how much they are speaking in percent)
            speaking_duration = self.speaking_duration.calculate_speaking_duration(speaker_overview)
        
            # PERMA score higher for teams that have a equal distribution of speaking time? (-> one score that indicates how equaly distributed they are speaking)
            # speaking_duration_share = self.calculate_speaking_duration_share(speaking_duration)
            
            speaking_duration_equality = self.speaking_duration.calculate_speaking_duration_equality(speaking_duration)
            print("Speaking duration equality (0 would be perfectly equal): ", speaking_duration_equality)
            self.speaking_duration.blocks_speaking_duration_equality["block"].append(block_id)
            self.speaking_duration.blocks_speaking_duration_equality["speaking_duration_equality"].append(speaking_duration_equality)
        
            #overlaps
            norm_num_overlaps = self.overlaps.calculate_amount_overlaps(speaker_overview, self.block_length[block_id])
            print("Number of overlaps (per minute per speaker): ", norm_num_overlaps)
            self.overlaps.blocks_norm_num_overlaps["block"].append(block_id)
            self.overlaps.blocks_norm_num_overlaps["norm_num_overlaps"].append(norm_num_overlaps)
            
            print("\n")
        
            # Visualize the communication patterns
        self.visualize_pattern(self.turn_taking.blocks_number_turns_equality, self.speaking_duration.blocks_speaking_duration_equality, self.overlaps.blocks_norm_num_overlaps)
            
    def get_rttm_path(self, video_name) -> str:
        
        rttm_folder_path = str(VIDEOS_DIR / video_name / "pyavi")
        rttm_file = glob.glob(os.path.join(rttm_folder_path, "*.rttm"))
        
        if not rttm_file:
            raise Exception("No rttm file found for path: " + rttm_folder_path)
        else:
            rttm_path = rttm_file[0]

        return rttm_path
        
        
        
    def visualize_pattern(self, blocks_number_turns_equality, blocks_speaking_duration_equality, blocks_norm_num_overlaps) -> None:
        
        # Create a figure and two subplots (one for each dictionary)
        fig, axes = plt.subplots(2, 2, figsize=(12, 9))

        # Plot the first dictionary
        axes[0,0].plot(blocks_number_turns_equality['block'], blocks_number_turns_equality['number_turns_equality'])
        axes[0,0].set_title('Equality (based on number of turns) per block \n - 0 is perfectly equal')
        axes[0,0].set_ylabel('Equality')

        # Plot the second dictionary
        axes[0,1].plot(blocks_speaking_duration_equality['block'], blocks_speaking_duration_equality['speaking_duration_equality'])
        axes[0,1].set_title('Equality (based on speaking duration) per block \n - 0 is perfectly equal')
        axes[0,1].set_ylabel('Equality')
        
        axes[1,0].plot(blocks_norm_num_overlaps['block'], blocks_norm_num_overlaps['norm_num_overlaps'])
        axes[1,0].set_title('Norm. number of overlaps per block \n - per minute per speaker')
        axes[1,0].set_ylabel('Norm. number of overlaps')
        
        plt.show()
            