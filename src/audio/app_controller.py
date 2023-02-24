# Here runs the overall pipeline of the audio processing

from src.audio.utils.rttm_file_preparation import RTTMFilePreparation
from src.audio.ASD.utils.asd_pipeline_tools import extract_audio_from_video

from src.audio.ASD.speaker_diar_pipeline import ASDSpeakerDirPipeline
from src.audio.com_pattern.com_pattern_analysis import ComPatternAnalysis
from src.audio.emotions.emotion_analysis import EmotionAnalysis

from src.audio.ASD.utils.asd_pipeline_tools import get_video_path
from src.audio.ASD.utils.asd_pipeline_tools import get_frames_per_second
from src.audio.ASD.utils.asd_pipeline_tools import get_num_total_frames

from src.audio.utils.analysis_tools import visualize_emotions, write_results_to_csv, visualize_com_pattern

from src.audio.utils.constants import VIDEOS_DIR

class Runner:
    def __init__(self, args):
        self.args = args
        self.run_pipeline_parts = args.get("RUN_PIPELINE_PARTS", [1,2])
        self.n_data_loader_thread = args.get("N_DATA_LOADER_THREAD",32)
        
        self.unit_of_analysis = args.get("UNIT_OF_ANALYSIS", 300)
        
        # Get video features
        self.video_name = args.get("VIDEO_NAME","001")
        self.video_path, self.save_path = get_video_path(self.video_name)
        self.num_frames_per_sec = get_frames_per_second(self.video_path)
        self.total_frames = get_num_total_frames(self.video_path)
        self.length_video = int(self.total_frames / self.num_frames_per_sec)
        
        # RTTM File Preparation
        self.rttm_file_preparation = RTTMFilePreparation(self.video_name, self.unit_of_analysis, self.length_video)
        
        # Extract audio from video (needed for several pipeline steps)
        audio_storage_folder = str(VIDEOS_DIR / self.video_name )
        self.audio_file_path = extract_audio_from_video(audio_storage_folder, self.video_path, self.n_data_loader_thread)

        # Path to the csv file with all the results
        csv_filename = self.video_name + "_audio_analysis_results.csv"
        self.csv_path = str(VIDEOS_DIR / self.video_name / csv_filename)
        
        # Initialize the parts of the pipelines
        self.asd_pipeline = ASDSpeakerDirPipeline(self.args, self.num_frames_per_sec, self.total_frames, self.audio_file_path)
        self.com_pattern_analysis = ComPatternAnalysis(self.video_name, self.unit_of_analysis)
        self.emotion_analysis = EmotionAnalysis(self.audio_file_path, self.unit_of_analysis)

    def run(self):
        
        # Perform combined Active Speaker Detection and Speaker Diarization - if selected in config file
        if 1 in self.run_pipeline_parts:
            self.asd_pipeline.run()

        # Calculate communication patterns based on the output of the ASD pipeline (rttm file) - if selected in config file
        if 2 in self.run_pipeline_parts:
            # Get the speaker overview and other data from the rttm file
            splitted_speaker_overview = self.rttm_file_preparation.read_rttm_file()
            # Based on the unit of analysis and the length of the video, create a list with the length of each block
            block_length = self.rttm_file_preparation.get_block_length()
            num_speakers = self.rttm_file_preparation.get("num_speakers")            
    
            # TODO: Nomalize com features (e.g. by time or max amount of turns)
            com_pattern_output = self.com_pattern_analysis.run(splitted_speaker_overview, block_length, num_speakers)

            emotions_output = self.emotion_analysis.run(splitted_speaker_overview)
        
            write_results_to_csv(emotions_output, com_pattern_output, self.csv_path, self.video_name)
            
        visualize_emotions(self.csv_path, self.unit_of_analysis, self.video_name)
        visualize_com_pattern(self.csv_path, self.unit_of_analysis, self.video_name)
            
        # Model inference here (training somewhere else)
        # when training the model, dann scaling ALL the values to the same range? 
        # that scaling has then also to be used for the inference
