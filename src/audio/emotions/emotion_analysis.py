import os
import audeer
import audonnx
import numpy as np
import torch
import math

from pydub import AudioSegment

from src.audio.utils.constants import EMOTIONS_DIR

class EmotionAnalysis:
    def __init__(self, audio_file_path, unit_of_analysis) -> None:
        
        self.audio_file_path = audio_file_path
        self.unit_of_analysis = unit_of_analysis
        
        model_name = 'model'
        self.model_root = os.path.join(EMOTIONS_DIR, model_name)   
        
        # TODO: to be tested, extracting did not work for me -> copy and paste URL into browser and download and extract manually
        #self.download_model() 
        
        # When GPU available, use 'cuda' instead of 'cpu'
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
            
        self.model = audonnx.load(self.model_root, device=self.device)
        
    def download_model(self):
        # * The entire download and extraction process
        
        cache_root = 'cache'

        audeer.mkdir(cache_root)
        def cache_path(file):
            return os.path.join(cache_root, file)

        url = 'https://zenodo.org/record/6221127/files/w2v2-L-robust-12.6bc4a7fd-1.1.0.zip'
        dst_path = cache_path('model.zip')

        if not os.path.exists(dst_path):
            audeer.download_url(url, dst_path, verbose=True)
            
        if not os.path.exists(self.model_root):
            audeer.extract_archive(dst_path, self.model_root, verbose=True)       

    def run(self, splitted_speaker_overview) -> None:
        
        # Create an empty list that has the length of splitted_speaker_overview
        emotions_output = [[] for i in range(len(splitted_speaker_overview))]

        
        audio_file = AudioSegment.from_wav(self.audio_file_path)
        sampling_rate = audio_file.frame_rate
        
        # For each block in splitted_speaker_overview, extract the audio based on the speaking segmetns and run the model
        for block_id, block in enumerate(splitted_speaker_overview):

            # print(" -- New Block -- ")
            # Loop through each speaker and append their audio segments to the concatenated_audio variable
            for speaker in block:
                speaker_id = speaker[0]
                start_times = speaker[1]
                end_times = speaker[2]
                speaker_audio = AudioSegment.empty()
                for i in range(len(start_times)):
                    start_time = start_times[i]*1000
                    end_time = end_times[i]*1000
                    speaker_audio += audio_file[start_time:end_time]
                
                
                # *For debugging: Save for each block and speaker the audio segment
                # audio_snippet_path = os.path.join(EMOTIONS_DIR, "audio_snippets" + "_s" + speaker_id + "_b" + str(block_id) + ".wav")
                # speaker_audio.export(audio_snippet_path, format="wav")
                
                output = self.get_audeer_emotions(speaker_audio, sampling_rate)
            
                # Logits order: arousal, dominance, valence.
                # arousal = output[0]
                # dominance = output[1]
                # valence = output[2]
                
                # Check if variable output is NaN, then set the emotions to the output
                # Otherwise set the emotions to NaN
                if not np.isnan(output).all():
                    arousal = output[0]
                    dominance = output[1]
                    valence = output[2]
                else:  
                    arousal = np.nan
                    dominance = np.nan
                    valence = np.nan
                
                # print("Speaker ID: ", speaker_id, "Arousal: ", arousal, "Dominance: ", dominance, "Valence: ", valence)
            
                # For each block, create a dictionary within the emotions_output list (where the key is the speaker_id and the value is a list of the emotions)
                emotions_output[block_id].append({speaker_id: [arousal, dominance, valence]})
                
            # Reformat the emotions_output
            emotions_output_reform = self.parse_emotions_output(emotions_output)

                
        return emotions_output_reform

    def parse_emotions_output(self, emotions_output) -> dict:
        emotions_output_reform = {}
        for block in emotions_output:
            for speaker_dict in block:
                speaker_id = list(speaker_dict.keys())[0]
                if speaker_id not in emotions_output_reform:
                    emotions_output_reform[speaker_id] = {'arousal': [], 'dominance': [], 'valence': []}
                values = speaker_dict[speaker_id]
                emotions_output_reform[speaker_id]['arousal'].append(values[0])
                emotions_output_reform[speaker_id]['dominance'].append(values[1])
                emotions_output_reform[speaker_id]['valence'].append(values[2])
                    
        return emotions_output_reform  

    def get_audeer_emotions(self, speaker_audio, sampling_rate) -> None:

        # It is max 5 min, because the model performs faster for shorter snippets (could be changed if using GPU for analysis)
        # TODO: maybe adapt in the future (based on paper they used 0.5s - ~35s snippets, see here https://arxiv.org/pdf/2203.07378v2.pdf)
        chunk_length_ms = min(self.unit_of_analysis*1000, 300000)

        # Calculate the number of chunks needed
        num_chunks = math.ceil(len(speaker_audio) / chunk_length_ms)
        
        chunk_outputs = []

        # Split the audio file into chunks
        for i in range(num_chunks):
            start = i * chunk_length_ms
            end = (i + 1) * chunk_length_ms
            chunk = speaker_audio[start:end]
            chunk = np.array(chunk.get_array_of_samples(), dtype=np.float32)
            #print(self.model(chunk, sampling_rate))
            chunk_outputs.append(self.model(chunk, sampling_rate)['logits'][0])

        # TODO: cutting the last chunk if it is shorter than the chunk_length_ms?
        
        # Calculate the average of the chunks
        output = np.mean(chunk_outputs, axis=0)

        
        return output


            
