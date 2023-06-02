from pathlib import Path

# Directories
root = Path.cwd()

CONFIG_DIR = root / "configs"
ASD_DIR = root / "src" / "audio" / "av_speaker_diarization"
VIDEOS_DIR = root / "src" / "audio" / "videos"
EMOTIONS_DIR = root / "src" / "audio" / "emotions"
PERMA_MODEL_DIR = root / "src" / "audio" / "perma_model"
PERMA_MODEL_RESULTS_DIR = root / "src" / "audio" / "perma_model" / "results"

EMOTION_MODEL_URL = 'https://zenodo.org/record/6221127/files/w2v2-L-robust-12.6bc4a7fd-1.1.0.zip'

FEATURE_NAMES = ['arousal', 'dominance', 'valence', 
                'norm_num_interruptions_absolute', 'norm_num_interruptions_relative', 
                'norm_num_utterances_absolute', 'norm_num_utterances_relative', 
                'norm_speak_duration_absolute', 'norm_speak_duration_relative']