from src.audio.app_controller import Runner
from src.audio.app_gui import parse_arguments

# TODO: create new requirements.txt, create readme etc.

def main() -> None:
    args = parse_arguments()
    video_path = '/Users/tobiaszeulner/Desktop/Master_Thesis_MIT/Teamwork-audio-analysis/src/audio/videos/001.mp4'
    runner = Runner(args, video_path)
    runner.run()


if __name__ == "__main__":
    main()