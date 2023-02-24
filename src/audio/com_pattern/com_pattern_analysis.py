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
            
            # TODO: Rename "unit" to "norm_speaking duration" for example and rename "team" to "team_share"?
            # TODO: further diff: absolute and relative
            # TODO: norm_overlaps_absolute, norm_overlaps_relative
            # TODO: show all relatives and all absolutes in one graph
            # TODO: go over the comments for each function (turns,...) and check if they are still correct
            # TODO: generalize relative and absolute functions (-> everywhere the same!)
            
            # * absolute = x per minute (for each speaker), e.g. 2.5 overlaps per minute
            # * relative = x in relation to team members (1 is avg., lower means under average, higher means above average)
            
            # Turn Taking features
            number_turns = self.turn_taking.calculate_number_turns(speaker_overview)
            norm_num_turns_absolute = self.turn_taking.calculate_norm_num_turns_absolute(number_turns, block_length[block_id])
            norm_num_turns_relative = self.turn_taking.calculate_norm_num_turns_relative(number_turns)
        
            # Speaking Durations features
            speaking_duration = self.speaking_duration.calculate_speaking_duration(speaker_overview)
            norm_speak_duration_absolute = self.speaking_duration.calculate_norm_speak_duration_absolute(speaking_duration, block_length[block_id])
            norm_speak_duration_relative = self.speaking_duration.calculate_norm_speak_duration_relative(speaking_duration)
        
            # Overlap features
            # * defined as: how often did I fall into a word of someone else (he/she is starting, then I have an overlap afterwards)
            num_overlaps = self.overlaps.calculate_number_overlaps(speaker_overview, block_length[block_id], block_id, num_speakers)
            norm_num_overlaps_absolute = self.overlaps.calculate_norm_num_overlaps_absolute(num_overlaps, block_length[block_id])
            norm_num_overlaps_relative = self.overlaps.calculate_norm_num_overlaps_relative(num_overlaps)
            
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