# Here runs the overall pipeline of the audio processing

from src.audio.ASD.speaker_diar_pipeline import ASDSpeakerDirPipeline
from src.audio.com_pattern.com_pattern_analysis import ComPatternAnalysis

from src.audio.ASD.utils.asd_pipeline_tools import get_video_path
from src.audio.ASD.utils.asd_pipeline_tools import get_frames_per_second
from src.audio.ASD.utils.asd_pipeline_tools import get_num_total_frames

class Runner:
    def __init__(self, args):
        self.args = args
        self.run_pipeline_parts = args.get("RUN_PIPELINE_PARTS", [1,2])
        
        # Calculate video length
        self.video_name = args.get("VIDEO_NAME","001")
        self.video_path, self.save_path = get_video_path(self.video_name)
        self.num_frames_per_sec = get_frames_per_second(self.video_path)
        self.total_frames = get_num_total_frames(self.video_path)
        self.length_video = int(self.total_frames / self.num_frames_per_sec)

    def run(self):
        
        # Perform combined Active Speaker Detection and Speaker Diarization - if selected in config file
        if 1 in self.run_pipeline_parts:
            self.asd_pipeline = ASDSpeakerDirPipeline(self.args)
            self.asd_pipeline.run()

        # TODO: make it more generell -> provide list of features what to calculate
        # Calculate communication patterns based on the output of the ASD pipeline (rttm file) - if selected in config file
        if 2 in self.run_pipeline_parts:
            self.com_pattern = ComPatternAnalysis(self.args, self.length_video)
            self.com_pattern.run()

