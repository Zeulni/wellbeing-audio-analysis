import statistics

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
            # TODO: go over the comments for each function (turns,...) and check if they are still correct
            
            # * absolute = x per minute (for each speaker), e.g. 2.5 overlaps per minute
            # * relative = x in relation to team members (1 is avg., lower means under average, higher means above average)
            
            # Turn Taking features
            number_turns = self.turn_taking.calculate_number_turns(speaker_overview)
            norm_num_turns_absolute = self.calc_norm_absolute_features(block_length[block_id], number_turns, "number_turns", "norm_num_turns_absolute")           
            norm_num_turns_relative = self.calc_norm_relative_features(number_turns, "number_turns", "norm_num_turns_relative")
        
            # Speaking Durations features
            speaking_duration = self.speaking_duration.calculate_speaking_duration(speaker_overview)
            norm_speak_duration_absolute = self.calc_norm_absolute_features(block_length[block_id], speaking_duration, "speaking_duration", "norm_speak_duration_absolute")
            norm_speak_duration_relative = self.calc_norm_relative_features(speaking_duration, "speaking_duration", "norm_speak_duration_relative")
        
            # Overlap features
            # * defined as: how often did I fall into a word of someone else (he/she is starting, then I have an overlap afterwards)
            num_overlaps = self.overlaps.calculate_number_overlaps(speaker_overview)
            norm_num_overlaps_absolute = self.calc_norm_absolute_features(block_length[block_id], num_overlaps, "num_overlaps", "norm_num_overlaps_absolute")
            norm_num_overlaps_relative = self.calc_norm_relative_features(num_overlaps, "num_overlaps", "norm_num_overlaps_relative")
            
            print("\n")
            
            # Loop through each speaker and then add the communication pattern features to the output list
            for speaker_id in number_turns["speaker"]:
                # Get the index of the speaker ID from the number_turns list
                speaker_id_index = number_turns["speaker"].index(speaker_id)
                # com_pattern_output[block_id].append({speaker_id: [norm_num_turns_relative["norm_num_turns_relative"][speaker_id_index], ind_speaking_shares_unit["ind_speaking_share_unit"][speaker_id_index], ind_speaking_shares_team["ind_speaking_share_team"][speaker_id_index]]})
                com_pattern_output[block_id].append({speaker_id: \
                    [norm_num_turns_absolute["norm_num_turns_absolute"][speaker_id_index], norm_num_turns_relative["norm_num_turns_relative"][speaker_id_index], \
                    norm_speak_duration_absolute["norm_speak_duration_absolute"][speaker_id_index], norm_speak_duration_relative["norm_speak_duration_relative"][speaker_id_index], \
                    norm_num_overlaps_absolute["norm_num_overlaps_absolute"][speaker_id_index], norm_num_overlaps_relative["norm_num_overlaps_relative"][speaker_id_index]]})
                print("Speaker ID: ", speaker_id)
                print("norm_num_turns_absolute: ", norm_num_turns_absolute["norm_num_turns_absolute"][speaker_id_index])
                print("norm_num_turns_relative: ", norm_num_turns_relative["norm_num_turns_relative"][speaker_id_index])
                print("norm_speak_duration_absolute: ", norm_speak_duration_absolute["norm_speak_duration_absolute"][speaker_id_index])
                print("norm_speak_duration_relative: ", norm_speak_duration_relative["norm_speak_duration_relative"][speaker_id_index])
                print("norm_num_overlaps_absolute: ", norm_num_overlaps_absolute["norm_num_overlaps_absolute"][speaker_id_index])
                print("norm_num_overlaps_relative: ", norm_num_overlaps_relative["norm_num_overlaps_relative"][speaker_id_index])
                print("\n")
                
        com_pattern_output_reform = self.parse_com_pattern_output(com_pattern_output)
        
        return com_pattern_output_reform
        
    def parse_com_pattern_output(self, com_pattern_output) -> dict:
        com_pattern_output_reform = {}
        for block in com_pattern_output:
            for speaker_dict in block:
                speaker_id = list(speaker_dict.keys())[0]
                if speaker_id not in com_pattern_output_reform:
                    com_pattern_output_reform[speaker_id] = {'norm_num_turns_absolute': [], 'norm_num_turns_relative': [], 'norm_speak_duration_absolute': [], \
                        'norm_speak_duration_relative': [], 'norm_num_overlaps_absolute': [], 'norm_num_overlaps_relative': []}
                values = speaker_dict[speaker_id]

                com_pattern_output_reform[speaker_id]['norm_num_turns_absolute'].append(values[0])
                com_pattern_output_reform[speaker_id]['norm_num_turns_relative'].append(values[1])
                com_pattern_output_reform[speaker_id]['norm_speak_duration_absolute'].append(values[2])
                com_pattern_output_reform[speaker_id]['norm_speak_duration_relative'].append(values[3])
                com_pattern_output_reform[speaker_id]['norm_num_overlaps_absolute'].append(values[4])
                com_pattern_output_reform[speaker_id]['norm_num_overlaps_relative'].append(values[5])
                    
        return com_pattern_output_reform  
    
    # Calculate x per minute (for each speaker)
    def calc_norm_absolute_features(self, block_length: int, from_feature_dict: dict, from_feature: str, to_feature: str) -> dict:
        norm_absolute_feature= {}
        
        norm_absolute_feature["speaker"] = []
        norm_absolute_feature[to_feature] = []
        
        # Divide by block length to get feature/sec, then multiply by 60 to get feature/min
        for speaker in from_feature_dict["speaker"]:
            norm_absolute_feature["speaker"].append(speaker)
            share = round((from_feature_dict[from_feature][from_feature_dict["speaker"].index(speaker)] / block_length)*60,3)
            norm_absolute_feature[to_feature].append(share)
            
        return norm_absolute_feature
    
    
    def calc_norm_relative_features(self, from_feature_dict: dict, from_feature: str, to_feature: str) -> dict:
        
        # Calculate the total value (e.g. total speaking duration over all speakers)
        total_sum_feature = sum(from_feature_dict[from_feature])
        
        # If there are no features at all, set it to 1 to avoid division by zero (results will be 0 anyway)
        if total_sum_feature == 0:
            total_sum_feature = 1
            
        norm_relative_feature = {}
        norm_relative_feature["speaker"] = []
        norm_relative_feature[to_feature] = []
        
        for speaker in from_feature_dict["speaker"]:
            norm_relative_feature["speaker"].append(speaker)
            share = round((from_feature_dict[from_feature][from_feature_dict["speaker"].index(speaker)] / total_sum_feature)*100,3)
            norm_relative_feature[to_feature].append(share)
            
        mean_share = statistics.mean(norm_relative_feature[to_feature])
        
        if mean_share == 0:
            mean_share = 1
            
        # Go trough the list entry again and divide by the mean share
        for speaker in norm_relative_feature["speaker"]:
            norm_relative_feature[to_feature][norm_relative_feature["speaker"].index(speaker)] = \
                round(norm_relative_feature[to_feature][norm_relative_feature["speaker"].index(speaker)] / mean_share,3)
                
        return norm_relative_feature