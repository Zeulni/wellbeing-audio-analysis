# import audeer
import audonnx
import numpy as np
import audinterface

from pydub import AudioSegment

from src.audio.utils.constants import EMOTIONS_DIR

# TODO: add library audonnx and audinterface??? to requirements.txt



import os

def run_test() -> None:
    # TODO: extracting did not work for me -> copy and paste URL into browser and download and extract manually
    # * The entire download and extraction process
    model_name = 'model'
    # cache_root = 'cache'


    # audeer.mkdir(cache_root)
    # def cache_path(file):
    #     return os.path.join(cache_root, file)


    # url = 'https://zenodo.org/record/6221127/files/w2v2-L-robust-12.6bc4a7fd-1.1.0.zip'
    # dst_path = cache_path('model.zip')

    # if not os.path.exists(dst_path):
    #     audeer.download_url(
    #         url, 
    #         dst_path, 
    #         verbose=True,
    #     )
        
    # if not os.path.exists(model_root):
    #     audeer.extract_archive(
    #         dst_path, 
    #         model_root, 
    #         verbose=True,
    #     )
        
    model_root = os.path.join(EMOTIONS_DIR, model_name)    
        
    model = audonnx.load(model_root)
    
    np.random.seed(0)

    sampling_rate = 16000
    # signal = np.random.normal(
    #     size=sampling_rate,
    # ).astype(np.float32)
    
    audio_file = os.path.join(EMOTIONS_DIR, 'test_long.wav')
    
    sound = AudioSegment.from_wav(audio_file)
    
    samplerate = sound.frame_rate
    trans_segment = np.array(sound.get_array_of_samples(), dtype=np.float32)

    #model(signal, sampling_rate)
    
    interface = audinterface.Feature(
    model.labels('logits'),
    process_func=model,
    process_func_args={
        'outputs': 'logits',
    },
    sampling_rate=sampling_rate,
    resample=True,    
    verbose=True,
    )
    
    # TODO: Files have to be in smaller chunks (RAM or memory overflow)
    print(interface.process_signal(trans_segment, sampling_rate))