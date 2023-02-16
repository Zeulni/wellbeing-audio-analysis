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
        # TODO: for every com pattern, create a new class
        self.com_pattern = ComPatternAnalysis(self.args, length_video)
        # Calculate communication patterns based on the output of the ASD pipeline (rttm file)
        self.com_pattern.run()

