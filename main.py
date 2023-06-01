from src.audio.app_controller import Runner
from src.audio.app_gui import parse_arguments

from src.audio.utils.calculate_time_series_features import CalculateTimeSeriesFeatures
from src.audio.perma_model.perma_model_training import PermaModelTraining

from src.audio.perma_model.perma_scores_dataset import run

# video_path is used to run with one call over multiple videos (for the normal usage this is not needed)
def main(video_path = None) -> None:
    args = parse_arguments()
    runner = Runner(args, video_path)
    runner.run()
    
    # perma_model_training = PermaModelTraining()
    # perma_model_training.run()

    # run()

if __name__ == "__main__":
    main()