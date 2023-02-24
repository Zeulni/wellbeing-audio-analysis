import statistics

class Overlaps:
    def __init__(self) -> None:
        pass
        
    def get(self, attribute):
        return getattr(self, attribute)
        
    def calculate_number_overlaps(self, speaker_overview, block_length, block_id, num_speakers) -> float:

        num_speakers = len(speaker_overview)
        
        # * In this matrix the information is captured who falls how often into the word of others
        overlaps_matrix = [[0 for _ in range(num_speakers)] for _ in range(num_speakers)]

        for i, speaker1 in enumerate(speaker_overview):
            for j, speaker2 in enumerate(speaker_overview):
                if i == j:
                    continue  # Skip comparison with self
                for start_time, end_time in zip(speaker2[1], speaker2[2]):
                    for start_time2, end_time2 in zip(speaker1[1], speaker1[2]):
                        if start_time2 <= end_time < end_time2 and start_time2 > start_time:
                            overlaps_matrix[i][j] += 1  # Count overlap for speaker i but not speaker j

        # for i, speaker_data_i in enumerate(speaker_overview):
        #     overlaps_matrix_i = [str(overlaps_matrix[i][j]) for j in range(num_speakers)]
        #     print(f"Speaker {speaker_data_i[0]} has {', '.join(overlaps_matrix_i)} overlaps with the other speakers")

        # Calculate per speaker the number of overlaps with the other speakers
        num_overlaps_list = [sum(overlaps_matrix[i]) for i in range(num_speakers)]
        
        num_overlaps = {}
        
        num_overlaps["speaker"] = []
        num_overlaps["num_overlaps"] = []

        for speaker in speaker_overview:
            num_overlaps["speaker"].append(speaker[0])
            # num overlaps has the same order as speaker_overview
            num_overlaps["num_overlaps"].append(num_overlaps_list[speaker_overview.index(speaker)])
        
        return num_overlaps
    
    # Calculate the number of overlaps per minute (for each speaker)
    def calculate_norm_num_overlaps_absolute(self, num_overlaps, block_length) -> dict:
        # Calculates the speaking duration of each speaker and saves it in a list 
        
        norm_num_overlaps_absolute = {}
        
        # Calculate the share in speaking time of each speaker
        norm_num_overlaps_absolute["speaker"] = []
        norm_num_overlaps_absolute["norm_num_overlaps_absolute"] = []
        
        for speaker in num_overlaps["speaker"]:
            norm_num_overlaps_absolute["speaker"].append(speaker)
            share = round((num_overlaps["num_overlaps"][num_overlaps["speaker"].index(speaker)] / block_length)*60,3)
            norm_num_overlaps_absolute["norm_num_overlaps_absolute"].append(share)
            
        return norm_num_overlaps_absolute
    
    # Calculates based on the speaking_duration dict the share in speaking time of each speaker    
    def calculate_norm_num_overlaps_relative(self, num_overlaps) -> dict:
        
        norm_num_overlaps_relative = {}
        
        # Calculate the total speaking duration
        total_num_overlaps = sum(num_overlaps["num_overlaps"])
        
        # If there are no overlaps at all, set it to 1 to avoid division by zero (results will be 0 anyway)
        if total_num_overlaps == 0:
            total_num_overlaps = 1
        
        # Calculate the share in speaking time of each speaker
        norm_num_overlaps_relative["speaker"] = []
        norm_num_overlaps_relative["norm_num_overlaps_relative"] = []
        
        for speaker in num_overlaps["speaker"]:
            norm_num_overlaps_relative["speaker"].append(speaker)
            share = (num_overlaps["num_overlaps"][num_overlaps["speaker"].index(speaker)] / total_num_overlaps)*100
            share = round(share, 1)
            norm_num_overlaps_relative["norm_num_overlaps_relative"].append(share)
            
        # Now dividing it by the average share of each speaker to normalize it
        mean_share = statistics.mean(norm_num_overlaps_relative["norm_num_overlaps_relative"])
        
        # If there are no overlaps at all, set it to 1 to avoid division by zero (results will be 0 anyway)
        if mean_share == 0:
            mean_share = 1

        # Go through each speaker and divide the share by the mean share
        for speaker in norm_num_overlaps_relative["speaker"]:
            norm_num_overlaps_relative["norm_num_overlaps_relative"][norm_num_overlaps_relative["speaker"].index(speaker)] = \
             round(norm_num_overlaps_relative["norm_num_overlaps_relative"][norm_num_overlaps_relative["speaker"].index(speaker)] / mean_share, 2)
            
        return norm_num_overlaps_relative