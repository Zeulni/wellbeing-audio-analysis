from src.audio.app_controller import Runner
from src.audio.app_gui import parse_arguments

# TODO: create new requirements.txt, create readme etc.

def main(video_path = None) -> None:
    args = parse_arguments()
    runner = Runner(args, video_path)
    runner.run()


if __name__ == "__main__":
    main()