class Overlaps:
    def __init__(self) -> None:
        # For each block store the normalized number of overlaps
        self.blocks_norm_num_overlaps = {}
        self.blocks_norm_num_overlaps["block"] = []
        self.blocks_norm_num_overlaps["norm_num_overlaps"] = []
        
    def get(self, attribute):
        return getattr(self, attribute)
        
    def calculate_amount_overlaps(self, speaker_overview, block_length, block_id, num_speakers) -> float:

        num_overlaps = 0

        # Iterate through the speaker list and compare speech segments
        for i in range(len(speaker_overview)):
            for j in range(i+1, len(speaker_overview)):
                # Iterate through all speech segments for speaker i
                for seg_i in range(len(speaker_overview[i][1])):
                    # Iterate through all speech segments for speaker j
                    for seg_j in range(len(speaker_overview[j][1])):
                        # Check if the speech segments overlap
                        if max(speaker_overview[i][1][seg_i], speaker_overview[j][1][seg_j]) < min(speaker_overview[i][2][seg_i], speaker_overview[j][2][seg_j]):
                            num_overlaps += 1
        
        # Normalize the number of overlaps bei the length of the audio snippet and the number of speakers
        # -> #overlaps per minute per speaker
        
        # TODO: current assumption: the number of speakers is constant and does not change during one meeting
        
        norm_num_overlaps = (num_overlaps / (block_length * num_speakers))*60
        
        self.blocks_norm_num_overlaps["block"].append(block_id)
        self.blocks_norm_num_overlaps["norm_num_overlaps"].append(norm_num_overlaps)
        
        # Print the number of overlaps
        return norm_num_overlaps
    
    