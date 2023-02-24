import statistics

class SpeakingDuration:
    def __init__(self) -> None:
        # For each block store the equality feature (based on the speaking duration)
        self.blocks_speaking_duration_equality = {}
        self.blocks_speaking_duration_equality["block"] = []
        self.blocks_speaking_duration_equality["speaking_duration_equality"] = []
        
    def get(self, attribute):
        return getattr(self, attribute)
    
    def calculate_speaking_duration(self, speaker_overview) -> dict:
        # Calculates the speaking duration of each speaker and saves it in a list 
        
        speaking_duration = {}
        
        speaking_duration["speaker"] = []
        speaking_duration["speaking_duration"] = []

        for speaker in speaker_overview:
            speaking_duration["speaker"].append(speaker[0])
            speaking_duration["speaking_duration"].append(round(sum(speaker[2]) - sum(speaker[1]),2))
            
        return speaking_duration
        
    def calculate_ind_speaking_share_unit(self, speaking_duration, block_length) -> dict:
        # Calculates the speaking duration of each speaker and saves it in a list 
        
        ind_speaking_shares_unit = {}
        
        # Calculate the share in speaking time of each speaker
        ind_speaking_shares_unit["speaker"] = []
        ind_speaking_shares_unit["ind_speaking_share_unit"] = []
        
        for speaker in speaking_duration["speaker"]:
            ind_speaking_shares_unit["speaker"].append(speaker)
            share = round((speaking_duration["speaking_duration"][speaking_duration["speaker"].index(speaker)] / block_length)*100,2)
            ind_speaking_shares_unit["ind_speaking_share_unit"].append(share)
            
        return ind_speaking_shares_unit
        
    # Calculates based on the speaking_duration dict the share in speaking time of each speaker    
    def calculate_ind_speaking_share_team(self, speaking_duration) -> dict:
        
        ind_speaking_shares_team = {}
        
        # Calculate the total speaking duration
        total_speaking_duration = sum(speaking_duration["speaking_duration"])
        
        # Calculate the share in speaking time of each speaker
        ind_speaking_shares_team["speaker"] = []
        ind_speaking_shares_team["ind_speaking_share_team"] = []
        
        for speaker in speaking_duration["speaker"]:
            ind_speaking_shares_team["speaker"].append(speaker)
            share = (speaking_duration["speaking_duration"][speaking_duration["speaker"].index(speaker)] / total_speaking_duration)*100
            share = round(share, 1)
            ind_speaking_shares_team["ind_speaking_share_team"].append(share)
            
        return ind_speaking_shares_team
        
    # Calculating the equality based on the speaking duration   
    def calculate_speaking_duration_equality(self, speaking_duration, block_id) -> float:
        
        # TODO: Independent of number of length (as normalized by mean), but also ind. of #speakers?
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