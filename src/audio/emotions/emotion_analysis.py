import os
import audeer
import audonnx
import numpy as np
import torch
import math

from pydub import AudioSegment

from src.audio.utils.constants import EMOTIONS_DIR

# TODO: add audinterface to requirements.txt???

class EmotionAnalysis:
    def __init__(self, audio_file_path) -> None:
        
        self.audio_file_path = audio_file_path
        
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
        
        emotions_output = []
        
        audio_file = AudioSegment.from_wav(self.audio_file_path)
        sampling_rate = audio_file.frame_rate
        
        # For each block in splitted_speaker_overview, extract the audio based on the speaking segmetns and run the model
        for block in splitted_speaker_overview:

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
            
                output = self.get_audeer_emotions(speaker_audio, sampling_rate)
            
                # Logits order: arousal, dominance, valence.
                arousal = output[0]
                dominance = output[1]
                valence = output[2]
                print("Speaker ID: ", speaker_id, "Arousal: ", arousal, "Dominance: ", dominance, "Valence: ", valence)
            
                # For each block, create a dictionary within the emotions_output list (where the key is the speaker_id and the value is a list of the emotions)
                emotions_output.append({speaker_id: [arousal, dominance, valence]})
                
        return emotions_output


    def get_audeer_emotions(self, speaker_audio, sampling_rate) -> None:

        # Set the chunk length in milliseconds (20 seconds) , 20, 22, 24, 35 for 2 min
        # TODO: change to 20s
        chunk_length_ms = 20000

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

        # TODO: calculate it in 20s snippets for a better performance
        # output = self.model(speaker_audio, sampling_rate)['logits'][0]
        
        # Calculate the average of the chunks
        output = np.mean(chunk_outputs, axis=0)
        
        return output


            
