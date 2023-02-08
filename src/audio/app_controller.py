# Here runs the overall pipeline of the audio processing

from src.audio.ASD.asd_pipeline import asd_pipeline

class Runner:
    def __init__(self, args):
        self.args = args

    def run(self):
        asd_pipeline(self.args)
