# Here runs the overall pipeline of the audio processing
import os

from src.audio.utils.rttm_file_preparation import RTTMFilePreparation
from src.audio.utils.logger import Logger
from src.audio.av_speaker_diarization.utils.asd_pipeline_tools import ASDPipelineTools

from src.audio.av_speaker_diarization.speaker_diar_pipeline import ASDSpeakerDirPipeline
from src.audio.com_pattern.com_pattern_analysis import ComPatternAnalysis
from src.audio.emotions.emotion_analysis import EmotionAnalysis
from src.audio.perma_model.perma_model_inferencing import PermaModelInferencing

from src.audio.utils.analysis_tools import visualize_emotions, write_results_to_csv, visualize_com_pattern


class Runner:
    def __init__(self, args, video_path = None):
        self.args = args
        self.run_pipeline_parts = args.get("RUN_PIPELINE_PARTS", [1,2])
        self.n_data_loader_thread = args.get("N_DATA_LOADER_THREAD",32)
        
        self.unit_of_analysis = args.get("UNIT_OF_ANALYSIS", 300)
        
        self.asd_pipeline_tools = ASDPipelineTools()
        
        # Get video features
        if video_path == None:
            self.video_name = args.get("VIDEO_NAME","001")
            self.video_path, self.save_path = self.asd_pipeline_tools.get_video_path(self.video_name)
        else:
            self.video_path = video_path
            self.video_name = os.path.splitext(os.path.basename(self.video_path))[0]
            self.save_dir = os.path.dirname(self.video_path)
            self.save_path = os.path.join(self.save_dir, self.video_name)
            
        # Save the results in this folder    
        if not os.path.exists(self.save_path): 
            os.makedirs(self.save_path)    
            
        # Initialize the logger
        log_file_name = self.save_path + "/audio_analysis_log.txt"
        self.logger = Logger(log_file_name)
        self.asd_pipeline_tools.set_logger(self.logger)
        self.logger.log("\n\n----- Audio analysis started for video: " + self.video_name + " -----\n")
            
        self.num_frames_per_sec = self.asd_pipeline_tools.get_frames_per_second(self.video_path)
        
        # If num_frames_per_sec is not 25, then create a copy of the video with 25 fps and update the video_path, video_name, and num_frames_per_sec
        if self.num_frames_per_sec != 25:
            self.video_path, self.video_name, self.num_frames_per_sec = self.asd_pipeline_tools.create_video_copy_25fps(self.video_path, self.video_name, self.save_path)
        
        self.total_frames = self.asd_pipeline_tools.get_num_total_frames(self.video_path)
        self.length_video = int(self.total_frames / self.num_frames_per_sec)
        
        # If the video length is shorter than 2x the unit of analysis, then set the unit of analysis to the half of the video length
        if self.length_video < 2 * self.unit_of_analysis:
            self.unit_of_analysis = int(self.length_video / 2)
            self.logger.log("Unit of analysis set to " + str(self.unit_of_analysis) + ", as otherwise only 1 data point.")
        
        # RTTM File Preparation
        self.rttm_file_preparation = RTTMFilePreparation(self.video_name, self.unit_of_analysis, self.length_video, self.save_path, self.asd_pipeline_tools)
        
        # Extract audio from video (needed for several pipeline steps)
        self.audio_file_path = self.asd_pipeline_tools.extract_audio_from_video(self.save_path, self.video_path, self.n_data_loader_thread, self.video_name)

        # Path to the csv file with all the results
        csv_filename = self.video_name + "_audio_analysis_results.csv"
        self.csv_path = os.path.join(self.save_path, csv_filename)
        
        self.faces_id_path = os.path.join(self.save_path, 'faces_id')
        
        # Initialize the parts of the pipelines
        self.asd_pipeline = ASDSpeakerDirPipeline(self.args, self.num_frames_per_sec, self.total_frames, self.audio_file_path, 
                                                  self.video_path, self.save_path, self.video_name, self.asd_pipeline_tools, self.faces_id_path)
        self.com_pattern_analysis = ComPatternAnalysis(self.video_name, self.unit_of_analysis)
        self.emotion_analysis = EmotionAnalysis(self.audio_file_path, self.unit_of_analysis)
        
        self.perma_scale = args.get("PERMA_SCALE", "norm")
        self.perma_model_inferencing = PermaModelInferencing(self.csv_path, self.save_path, self.faces_id_path, self.perma_scale, self.logger)
        
    # Closing the logfile when the object is deleted
    def __del__(self):
        self.logger.close()

    def run(self):
        
        try:
            # Perform audiovisual speaker diarization
            if 1 in self.run_pipeline_parts:
                self.asd_pipeline.run()

            # Calculate communication patterns and emotions based on the rttm and audio file
            if 2 in self.run_pipeline_parts:
                # Get the speaker overview and other data from the rttm file
                splitted_speaker_overview = self.rttm_file_preparation.read_rttm_file(self.faces_id_path)
                # Based on the unit of analysis and the length of the video, create a list with the length of each block
                block_length = self.rttm_file_preparation.get_block_length()
                
                num_speakers = self.rttm_file_preparation.get("num_speakers")            
        
                com_pattern_output = self.com_pattern_analysis.run(splitted_speaker_overview, block_length, num_speakers)
                self.asd_pipeline_tools.write_to_terminal("Communication pattern analysis finished for " + str((len(com_pattern_output))) + " speakers")
                emotions_output = self.emotion_analysis.run(splitted_speaker_overview)
                self.asd_pipeline_tools.write_to_terminal("Emotion analysis finished for " + str(len(emotions_output)) + " speakers")
                write_results_to_csv(emotions_output, com_pattern_output, self.csv_path, self.video_name, self.asd_pipeline_tools)
                self.logger.log("Time series data written to csv file: " + self.csv_path)
                
            # Visualize the results
            if 3 in self.run_pipeline_parts:    
                visualize_emotions(self.csv_path, self.unit_of_analysis, self.video_name)
                visualize_com_pattern(self.csv_path, self.unit_of_analysis, self.video_name, ['norm_num_utterances_relative', 'norm_speak_duration_relative', 'norm_num_interruptions_relative'])
                visualize_com_pattern(self.csv_path, self.unit_of_analysis, self.video_name, ['norm_num_utterances_absolute', 'norm_speak_duration_absolute', 'norm_num_interruptions_absolute'])
                
            # PERMA model inference
            if 4 in self.run_pipeline_parts:
                self.perma_model_inferencing.run()
                
            self.asd_pipeline_tools.write_to_terminal("------- Pipeline finished successfully -------\n")
            
        except Exception as e:
            self.logger.log(str(e), level="ERROR")
            raise e
