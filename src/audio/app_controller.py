# Here runs the overall pipeline of the audio processing

from src.audio.ASD.speaker_diar_pipeline import ASDSpeakerDirPipeline

class Runner:
    def __init__(self, args):
        self.args = args

    def run(self):
        
        # Perform combined Active Speaker Detection and Speaker Diarization
        asd_pipeline = ASDSpeakerDirPipeline(self.args)
        asd_pipeline.run()
