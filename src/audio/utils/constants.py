from pathlib import Path

# Directories
root = Path.cwd()

CONFIG_DIR = root / "configs"
ASD_DIR = root / "src" / "audio" / "ASD"
VIDEOS_DIR = root / "src" / "audio" / "videos"
EMOTIONS_DIR = root / "src" / "audio" / "emotions"
PERMA_MODEL_DIR = root / "src" / "audio" / "perma_model"

FEATURE_NAMES = ['arousal', 'dominance', 'valence', 
                'norm_num_overlaps_absolute', 'norm_num_overlaps_relative', 
                'norm_num_turns_absolute', 'norm_num_turns_relative', 
                'norm_speak_duration_absolute', 'norm_speak_duration_relative']