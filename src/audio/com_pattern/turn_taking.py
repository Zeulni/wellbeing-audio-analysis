import statistics

class TurnTaking:
    def __init__(self) -> None:
        # For each block store the equality feature (based on the number of turns)
        self.blocks_number_turns_equality = {}
        self.blocks_number_turns_equality["block"] = []
        self.blocks_number_turns_equality["number_turns_equality"] = []
        
    def get(self, attribute):
        return getattr(self, attribute)
    
    def calculate_number_turns(self, speaker_overview) -> dict:
        # Calculates the number of turns (number of times each speakers starts a speaking turn) and saves it in a dict 
        
        number_turns = {}
        
        number_turns["speaker"] = []
        number_turns["number_turns"] = []
        
        for speaker in speaker_overview:
            number_turns["speaker"].append(speaker[0])
            number_turns["number_turns"].append(len(speaker[1]))
            
        return number_turns
        
    # Calculate based on the number_turns dict the share in number of turns of each speaker
    def calculate_number_turns_share(self, number_turns) -> dict:
        
        # Calculate the total number of turns
        total_number_turns = sum(number_turns["number_turns"])
        
        # Initialize the com pattern dictionaries
        number_turns_share = {}
        
        # Calculate the share in number of turns of each speaker
        number_turns_share["speaker"] = []
        number_turns_share["share_number_turns"] = []
        
        for speaker in number_turns["speaker"]:
            number_turns_share["speaker"].append(speaker)
            share = (number_turns["number_turns"][number_turns["speaker"].index(speaker)] / total_number_turns)*100
            share = round(share, 1)
            number_turns_share["share_number_turns"].append(share)
            
        return number_turns_share
        
    # Calculating the equality based on the number of turns  
    def calculate_number_turns_equality(self, number_turns, block_id) -> float:
        
        # TODO: Independent of number of length (as normalized by mean), but also ind. of #speakers?
        mean_time = statistics.mean(number_turns["number_turns"])
        stdev_time = statistics.stdev(number_turns["number_turns"])
        cv = (stdev_time / mean_time) * 100
        
        # TODO: Formula see Ignacio's thesis
        # Calculating the speaker equality under the assumption, that the max speaker duration is a good proxy for the 
        # expected speaking duration that each member should take in a perfectly equal distribution
        # max_cv = (math.sqrt(self.num_speakers) - 1) / math.sqrt(self.num_speakers)
        # speaker_equality = 1 - (cv / max_cv)
        
        self.blocks_number_turns_equality["block"].append(block_id)
        self.blocks_number_turns_equality["number_turns_equality"].append(cv)

        return cv       