class Overlaps:
    def __init__(self) -> None:
        pass
        
    def get(self, attribute):
        return getattr(self, attribute)
    
    # defined as: how often did I fall into a word of someone else (he/she is starting, then I have an overlap afterwards)    
    def calculate_number_overlaps(self, speaker_overview) -> float:

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