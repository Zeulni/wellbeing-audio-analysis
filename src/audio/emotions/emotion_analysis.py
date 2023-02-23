import os
import audeer
import audonnx
import numpy as np
import audinterface
import torch
import math

from pydub import AudioSegment

from src.audio.utils.constants import EMOTIONS_DIR

# TODO: add audinterface to requirements.txt???

class EmotionAnalysis:
    def __init__(self) -> None:
        
        model_name = 'model'
        self.model_root = os.path.join(EMOTIONS_DIR, model_name)   
        
        # TODO: to be tested, extracting did not work for me -> copy and paste URL into browser and download and extract manually
        #self.download_model() 
        
        # When GPU available, use 'cuda' instead of 'cpu'
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        
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

    def run(self) -> None:
    
        model = audonnx.load(self.model_root, device=self.device)
        
        audio_file = os.path.join(EMOTIONS_DIR, 'test_long.wav')
        
        sound = AudioSegment.from_wav(audio_file)
        sampling_rate = sound.frame_rate
        
        # Set the chunk length in milliseconds (20 seconds) , 20, 22, 24, 35 for 2 min
        chunk_length_ms = 20000

        # Calculate the number of chunks needed
        num_chunks = math.ceil(len(sound) / chunk_length_ms)

        # Split the audio file into chunks
        for i in range(num_chunks):
            start = i * chunk_length_ms
            end = (i + 1) * chunk_length_ms
            chunk = sound[start:end]
            chunk = np.array(chunk.get_array_of_samples(), dtype=np.float32)
            print(model(chunk, sampling_rate))

    
   #  samplerate = sound.frame_rate
    
    
    # Cut the the sound file into 20 second chunks and run the model on each chunk


    # print(model(trans_segment, sampling_rate))
    
    # interface = audinterface.Feature(
    # model.labels('logits'),
    # process_func=model,
    # process_func_args={
    #     'outputs': 'logits',
    # },
    # sampling_rate=sampling_rate,
    # resample=True,    
    # verbose=True,
    # )
    
    # # TODO: Files have to be in smaller chunks (RAM or memory overflow) 20 min for 4 min audio -> 5x on CPU
    
    # TODO: 20 sec video ging verhältnismäßig schnell!! alles einfach kleiner machen?
    # print(interface.process_signal(trans_segment, sampling_rate))