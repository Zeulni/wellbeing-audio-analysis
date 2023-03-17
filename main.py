from src.audio.app_controller import Runner
from src.audio.app_gui import parse_arguments

from src.audio.utils.calculate_time_series_features import CalculateTimeSeriesFeatures
from src.audio.perma_model.perma_model import PermaModel

# TODO: create new requirements.txt, create readme etc.

# video_path is used to run with one call over multiple videos (for the normal usage this is not needed)
def main(video_path = None) -> None:
    # args = parse_arguments()
    # runner = Runner(args, video_path)
    # runner.run()

    # calculate_features = CalculateTimeSeriesFeatures()
    # team = 'team_09_check'
    # calculate_features.run(team)
    
    perma_model = PermaModel()
    perma_model.run()

if __name__ == "__main__":
    main()