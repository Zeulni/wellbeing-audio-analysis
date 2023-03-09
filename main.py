from src.audio.app_controller import Runner
from src.audio.app_gui import parse_arguments

from src.audio.perma_model import PermaModel

# TODO: create new requirements.txt, create readme etc.

# video_path is used to run with one call over multiple videos (for the normal usage this is not needed)
def main(video_path = None) -> None:
    # args = parse_arguments()
    # runner = Runner(args, video_path)
    # runner.run()

    perma_model = PermaModel()
    perma_model.calculate_features()

if __name__ == "__main__":
    main()