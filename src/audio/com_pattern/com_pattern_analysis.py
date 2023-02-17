import matplotlib.pyplot as plt

from src.audio.utils.rttm_file_preparation import RTTMFilePreparation
from src.audio.com_pattern.turn_taking import TurnTaking
from src.audio.com_pattern.speaking_duration import SpeakingDuration
from src.audio.com_pattern.overlaps import Overlaps

# Perform certain communication pattern evaluations on the rttm file
class ComPatternAnalysis:
    def __init__(self, args, length_video) -> None:
        
        self.rttm_file_preparation = RTTMFilePreparation(args.get("VIDEO_NAME","001"), args.get("UNIT_OF_ANALYSIS", 300), length_video)
        
        self.splitted_speaker_overview = self.rttm_file_preparation.read_rttm_file()
        
        # Based on the unit of analysis and the length of the video, create a list with the length of each block
        self.block_length = self.rttm_file_preparation.get_block_length()
        
        # Initialize the communication pattern classes
        self.turn_taking = TurnTaking()
        self.speaking_duration = SpeakingDuration()
        self.overlaps = Overlaps(self.rttm_file_preparation.get("num_speakers"))
   
            
    def run(self) -> None:
        
        # For each unit of analysis (block) perform the following calculations
        for block_id, speaker_overview in enumerate(self.splitted_speaker_overview):
            # PERMA score higher for teams that start more conversations (e.g. shorter ones)
            number_turns = self.turn_taking.calculate_number_turns(speaker_overview)

            # PERMA score higher for teams that have a equal distribution of turns?
            # number_turns_share = self.turn_taking.calculate_number_turns_share(number_turns)

            number_turns_equality = self.turn_taking.calculate_number_turns_equality(number_turns, block_id)
            print("Number turns equality (0 would be perfectly equal): ", number_turns_equality)
        
            # PERMA score higher for teams that speak more? (-> calculate one score that indicates how much they are speaking in percent)
            speaking_duration = self.speaking_duration.calculate_speaking_duration(speaker_overview)
        
            # PERMA score higher for teams that have a equal distribution of speaking time? (-> one score that indicates how equaly distributed they are speaking)
            # speaking_duration_share = self.calculate_speaking_duration_share(speaking_duration)
            
            speaking_duration_equality = self.speaking_duration.calculate_speaking_duration_equality(speaking_duration, block_id)
            print("Speaking duration equality (0 would be perfectly equal): ", speaking_duration_equality)
        
            #overlaps
            norm_num_overlaps = self.overlaps.calculate_amount_overlaps(speaker_overview, self.block_length[block_id], block_id)
            print("Number of overlaps (per minute per speaker): ", norm_num_overlaps)
            
            print("\n")
        
            # Visualize the communication patterns
        self.visualize_pattern(self.turn_taking.get("blocks_number_turns_equality"), self.speaking_duration.get("blocks_speaking_duration_equality"), self.overlaps.get("blocks_norm_num_overlaps"))
        
        
        
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
            