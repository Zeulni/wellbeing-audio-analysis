# Here runs the overall pipeline of the audio processing

from src.audio.ASD.speaker_diar_pipeline import ASDSpeakerDirPipeline
from src.audio.com_pattern.com_pattern_analysis import ComPatternAnalysis

class Runner:
    def __init__(self, args):
        self.args = args

    def run(self):
        
        # Perform combined Active Speaker Detection and Speaker Diarization
        self.asd_pipeline = ASDSpeakerDirPipeline(self.args)
        length_video = self.asd_pipeline.run()

        # TODO: Make it more general (factory pattern per communication pattern?) - e.g. provide a list of communication patterns to be calculated
        self.com_pattern = ComPatternAnalysis(self.args, length_video)
        # Calculate communication patterns based on the output of the ASD pipeline (rttm file)
        
        # PERMA score higher for teams that start more conversations (e.g. shorter ones)
        self.com_pattern.calculate_number_turns()
        
        # PERMA score higher for teams that have a equal distribution of turns?
        self.com_pattern.calculate_number_turns_share()
        self.com_pattern.calculate_number_turns_equality()
        
        # PERMA score higher for teams that speak more? (-> calculate one score that indicates how much they are speaking in percent)
        self.com_pattern.calculate_speaking_duration()
        
        # PERMA score higher for teams that have a equal distribution of speaking time? (-> one score that indicates how equaly distributed they are speaking)
        self.com_pattern.calculate_speaking_duration_share()
        self.com_pattern.calculate_speaking_duration_equality()
        
        #overlaps
        self.com_pattern.calculate_amount_overlaps()
        
        
        
        self.com_pattern.visualize_pattern()
