class Overlaps:
    def __init__(self) -> None:
        # For each block store the normalized number of overlaps
        # self.blocks_norm_num_overlaps = {}
        # self.blocks_norm_num_overlaps["block"] = []
        # self.blocks_norm_num_overlaps["norm_num_overlaps"] = []
        pass
        
    def get(self, attribute):
        return getattr(self, attribute)
        
    def calculate_amount_overlaps(self, speaker_overview, block_length, block_id, num_speakers) -> float:

        num_speakers = len(speaker_overview)
        overlaps = [[0 for _ in range(num_speakers)] for _ in range(num_speakers)]

        for i, speaker1 in enumerate(speaker_overview):
            for j, speaker2 in enumerate(speaker_overview):
                if i == j:
                    continue  # Skip comparison with self
                for start_time, end_time in zip(speaker2[1], speaker2[2]):
                    for start_time2, end_time2 in zip(speaker1[1], speaker1[2]):
                        if start_time2 <= end_time < end_time2 and start_time2 > start_time:
                            overlaps[i][j] += 1  # Count overlap for speaker i but not speaker j

        for i, speaker_data_i in enumerate(speaker_overview):
            overlaps_i = [str(overlaps[i][j]) for j in range(num_speakers)]
            print(f"Speaker {speaker_data_i[0]} has {', '.join(overlaps_i)} overlaps with the other speakers")

        # Calculate per speaker the number of overlaps with the other speakers
        num_overlaps_list = [sum(overlaps[i]) for i in range(num_speakers)]
        
        num_overlaps = {}
        
        num_overlaps["speaker"] = []
        num_overlaps["num_overlaps"] = []

        for speaker in speaker_overview:
            num_overlaps["speaker"].append(speaker[0])
            # num overlaps has the same order as speaker_overview
            num_overlaps["num_overlaps"].append(num_overlaps_list[speaker_overview.index(speaker)])
            

        # Calculate the normalized number of overlaps
        
        # TODO: rename function?
        
        
        
        return num_overlaps
    
    