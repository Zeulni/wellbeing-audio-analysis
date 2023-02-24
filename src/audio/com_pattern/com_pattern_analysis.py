from src.audio.utils.analysis_tools import write_results_to_csv, visualize_individual_speaking_shares

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
        
        com_pattern_output = [[] for i in range(len(splitted_speaker_overview))]
        
        # For each unit of analysis (block) perform the following calculations
        for block_id, speaker_overview in enumerate(splitted_speaker_overview):
            
            # TODO: not the best feature for my use case, as some times one turn is recognized as mutliple turns instead
            # PERMA score higher for teams that start more conversations (e.g. shorter ones)
            number_turns = self.turn_taking.calculate_number_turns(speaker_overview)

            # PERMA score higher for teams that have a equal distribution of turns?
            # number_turns_share = self.turn_taking.calculate_number_turns_share(number_turns)

            number_turns_equality = self.turn_taking.calculate_number_turns_equality(number_turns, block_id)
            # print("Number turns equality (0 would be perfectly equal): ", number_turns_equality)
        
            # Speaking Durations
            speaking_duration = self.speaking_duration.calculate_speaking_duration(speaker_overview)
            ind_speaking_shares_unit = self.speaking_duration.calculate_ind_speaking_share_unit(speaking_duration, block_length[block_id])
        
            # PERMA score higher for teams that have a equal distribution of speaking time? (-> one score that indicates how equaly distributed they are speaking)
            ind_speaking_shares_team = self.speaking_duration.calculate_ind_speaking_share_team(speaking_duration)
            
            # speaking_duration_equality = self.speaking_duration.calculate_speaking_duration_equality(speaking_duration_share, block_id)
            # print("Speaking duration equality (0 would be perfectly equal): ", speaking_duration_equality)
        
            #overlaps
            norm_num_overlaps = self.overlaps.calculate_amount_overlaps(speaker_overview, block_length[block_id], block_id, num_speakers)
            # print("Number of overlaps (per minute per speaker): ", norm_num_overlaps)
            
            print("\n")
            
            #com_pattern_output[block_id].append({speaker_id: [arousal, dominance, valence]})
            
            # Loop through each speaker and then add number_turns, speaking_duration, norm_num_overlaps to the com_pattern_output (for each block there should be one list, within each list there should be a dict with the speaker ID as the key and the values as a list)
            for speaker_id in number_turns["speaker"]:
                
                # Get the index of the speaker ID from the number_turns list
                speaker_id_index = number_turns["speaker"].index(speaker_id)
                
                com_pattern_output[block_id].append({speaker_id: [number_turns["number_turns"][speaker_id_index], ind_speaking_shares_unit["ind_speaking_share_unit"][speaker_id_index], ind_speaking_shares_team["ind_speaking_share_team"][speaker_id_index]]})
        
                # print("Speaker ID: ", speaker_id, "Arousal: ", arousal, "Dominance: ", dominance, "Valence: ", valence)
                print("Speaker ID: ", speaker_id, "Number of turns: ", number_turns["number_turns"][speaker_id_index], "Ind. Speaking Share (Unit): ", ind_speaking_shares_unit["ind_speaking_share_unit"][speaker_id_index])
                
        # Write results to a csv file
        # csv_path = write_results_to_csv(self.turn_taking, self.speaking_duration, self.overlaps, self.video_name)
        
        com_pattern_output_reform = self.parse_emotions_output(com_pattern_output)
        
        return com_pattern_output_reform
        
    def parse_emotions_output(self, com_pattern_output) -> dict:
        com_pattern_output_reform = {}
        for block in com_pattern_output:
            for speaker_dict in block:
                speaker_id = list(speaker_dict.keys())[0]
                if speaker_id not in com_pattern_output_reform:
                    com_pattern_output_reform[speaker_id] = {'number_turns': [], 'ind_speaking_share_unit': [], 'ind_speaking_share_team': []}
                values = speaker_dict[speaker_id]
                com_pattern_output_reform[speaker_id]['number_turns'].append(values[0])
                com_pattern_output_reform[speaker_id]['ind_speaking_share_unit'].append(values[1])
                com_pattern_output_reform[speaker_id]['ind_speaking_share_team'].append(values[2])
                    
        return com_pattern_output_reform  