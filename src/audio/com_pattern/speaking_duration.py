import statistics

class SpeakingDuration:
    def __init__(self) -> None:
        # For each block store the equality feature (based on the speaking duration)
        self.blocks_speaking_duration_equality = {}
        self.blocks_speaking_duration_equality["block"] = []
        self.blocks_speaking_duration_equality["speaking_duration_equality"] = []
        
    def get(self, attribute):
        return getattr(self, attribute)
    
    # Calculates the speaking duration of each speaker (in seconds)
    def calculate_speaking_duration(self, speaker_overview) -> dict:  
        
        speaking_duration = {}
        
        speaking_duration["speaker"] = []
        speaking_duration["speaking_duration"] = []

        for speaker in speaker_overview:
            speaking_duration["speaker"].append(speaker[0])
            speaking_duration["speaking_duration"].append(round(sum(speaker[2]) - sum(speaker[1]),2))
            
        return speaking_duration
        
    # Calculating the equality based on the speaking duration   
    def calculate_speaking_duration_equality(self, speaking_duration, block_id) -> float:
        
        # Independent of number of length (as normalized by mean), but also ind. of #speakers?
        mean_time = statistics.mean(speaking_duration["ind_speaking_share"])
        stdev_time = statistics.stdev(speaking_duration["ind_speaking_share"])
        cv = (stdev_time / mean_time) * 100
        
        # TODO: Formula see Ignacio's thesis
        # Calculating the speaker equality under the assumption, that the max speaker duration is a good proxy for the 
        # expected speaking duration that each member should take in a perfectly equal distribution
        # max_cv = (math.sqrt(self.num_speakers) - 1) / math.sqrt(self.num_speakers)
        # speaker_equality = 1 - (cv / max_cv)
        
        self.blocks_speaking_duration_equality["block"].append(block_id)
        self.blocks_speaking_duration_equality["speaking_duration_equality"].append(cv)

        return cv