from src.audio.utils.analysis_tools import write_results_to_csv, visualize_pattern, visualize_individual_speaking_shares

from src.audio.com_pattern.turn_taking import TurnTaking
from src.audio.com_pattern.speaking_duration import SpeakingDuration
from src.audio.com_pattern.overlaps import Overlaps


# Perform certain communication pattern evaluations on the rttm file
class ComPatternAnalysis:
    def __init__(self, video_name, unit_of_analysis) -> None:
        
        self.video_name = video_name
        self.unit_of_analysis = unit_of_analysis
        
        # Initialize the communication pattern classes
        self.turn_taking = TurnTaking()
        self.speaking_duration = SpeakingDuration()
        self.overlaps = Overlaps()
            
    def run(self, splitted_speaker_overview, block_length, num_speakers) -> None:
        
        # For each unit of analysis (block) perform the following calculations
        for block_id, speaker_overview in enumerate(splitted_speaker_overview):
            
            # TODO: not the best feature for my use case, as some times one turn is recognized as mutliple turns instead
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
            norm_num_overlaps = self.overlaps.calculate_amount_overlaps(speaker_overview, block_length[block_id], block_id, num_speakers)
            print("Number of overlaps (per minute per speaker): ", norm_num_overlaps)
            
            print("\n")
        
        # Write results to a csv file
        csv_path = write_results_to_csv(self.turn_taking, self.speaking_duration, self.overlaps, self.video_name)
        
        # Visualize the communication patterns
        visualize_pattern(csv_path, self.unit_of_analysis, self.video_name)
        
        # Visualizes only the speaking duration overview of the last block
        # visualize_individual_speaking_shares(speaking_duration)

        
            