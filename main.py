from src.audio.app_controller import Runner
from src.audio.app_gui import parse_arguments

from src.audio.utils.calculate_time_series_features import CalculateTimeSeriesFeatures
from src.audio.perma_model.perma_model_training import PermaModelTraining

def main(video_path = None) -> None:
    args = parse_arguments()
    runner = Runner(args, video_path)
    runner.run()

if __name__ == "__main__":
    main()