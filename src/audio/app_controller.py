# Here runs the overall pipeline of the audio processing

from src.audio.ASD.speaker_diar_pipeline import ASDPipeline

class Runner:
    def __init__(self, args):
        self.args = args

    def run(self):
        asd_pipeline = ASDPipeline(self.args)
        asd_pipeline.run()
