import statistics

class Utterances:
    def __init__(self) -> None:
        # For each block store the equality feature (based on the number of utterances)
        self.blocks_number_utterances_equality = {}
        self.blocks_number_utterances_equality["block"] = []
        self.blocks_number_utterances_equality["number_utterances_equality"] = []
        
    def get(self, attribute):
        return getattr(self, attribute)
    
    # Calculates the number of utterances (number of times each speakers starts a speaking utterance)
    def calculate_number_utterances(self, speaker_overview) -> dict:
        
        number_utterances = {}
        
        number_utterances["speaker"] = []
        number_utterances["number_utterances"] = []
        
        for speaker in speaker_overview:
            number_utterances["speaker"].append(speaker[0])
            number_utterances["number_utterances"].append(len(speaker[1]))
            
        return number_utterances
        
    # Team Feature: Calculating the equality based on the number of utterances  
    def calculate_number_utterances_equality(self, number_utterances, block_id) -> float:
        
        # TODO: Independent of number of length (as normalized by mean), but also ind. of #speakers?
        mean_time = statistics.mean(number_utterances["number_utterances"])
        stdev_time = statistics.stdev(number_utterances["number_utterances"])
        cv = (stdev_time / mean_time) * 100
        
        # TODO: Formula see Ignacio's thesis
        # Calculating the speaker equality under the assumption, that the max speaker duration is a good proxy for the 
        # expected speaking duration that each member should take in a perfectly equal distribution
        # max_cv = (math.sqrt(self.num_speakers) - 1) / math.sqrt(self.num_speakers)
        # speaker_equality = 1 - (cv / max_cv)
        
        self.blocks_number_utterances_equality["block"].append(block_id)
        self.blocks_number_utterances_equality["number_utterances_equality"].append(cv)

        return cv       