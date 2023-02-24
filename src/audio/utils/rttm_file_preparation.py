import glob
import os

from src.audio.utils.constants import VIDEOS_DIR

class RTTMFilePreparation:
    def __init__(self, video_name, unit_of_analysis, length_video) -> None:
        self.video_name = video_name
        self.unit_of_analysis = unit_of_analysis
        self.length_video = length_video
        
        # TODO: Testing
        # self.length_video = 100
        
        self.num_speakers = None
        
    def get(self, attribute):
        return getattr(self, attribute)
    
    def get_rttm_path(self) -> str:
        
        rttm_folder_path = str(VIDEOS_DIR / self.video_name)
        rttm_file = glob.glob(os.path.join(rttm_folder_path, "*.rttm"))
        
        if not rttm_file:
            raise Exception("No rttm file found for path: " + rttm_folder_path)
        else:
            rttm_path = rttm_file[0]

        return rttm_path
    
    def read_rttm_file(self) -> list:
        
        rttm_file_path = self.get_rttm_path()
        
        # Read rttm file
        self.myfile = open(rttm_file_path, "r")
        
        start_time = []
        end_time = []
        speaker_id = []

        # store speaker ID, start time and end time in a list
        for line in self.myfile:
            split_line = line.split()

            # creating 3 lists (speaker, start, end)
            # get start and end time in seconds
            start_time.append(round(float(split_line[3]),2))
            end_time.append(round(float(split_line[3]) + float(split_line[4]),2))
            speaker_id.append(split_line[7])
            
        # find unique speaker IDs
        unique_speaker_id = list(set(speaker_id))
        
        # Number of speakers
        self.num_speakers = len(unique_speaker_id)
        
        # List including all start and end times of all speakers
        self.speaker_overview = []

        # find indices of unique speaker IDs, get start and end time of each speaker and concatenate all audio segments of the same speaker
        for speaker in unique_speaker_id:
            # find indices of speaker
            indices = [i for i, x in enumerate(speaker_id) if x == speaker]

            # get start and end time of speaker
            start_time_speaker = [start_time[i] for i in indices]
            end_time_speaker = [end_time[i] for i in indices]
            
            # Make sure that the start and end times are sorted
            start_time_speaker.sort()
            end_time_speaker.sort()
            
            self.speaker_overview.append([speaker, start_time_speaker, end_time_speaker])
            
        
        # Sort the speaker overview based on the speaker_id
        self.speaker_overview.sort(key=lambda x: x[0])
        
        self.user_input_handling()
        
        # TODO: Testing
        # self.speaker_overview = [['0', [0, 90], [10, 100]], ['1', [5], [15]], ['2', [50], [80]], ['3', [75], [95]]]
        
        
        self.block_speaker_overview = self.split_speaker_overview()
        
        return self.block_speaker_overview
    
    def user_input_handling(self) -> None:
        # Ask the user if he wants to remove a speaker (if he wants to remove a speaker, he should enter the speaker ID)
        # Once that speaker was removed, ask again (until the user enters nothing)
        while True: 
            # Print the list of speakers
            print("The following speakers were detected:", end=" ")
            for speaker_id, speaker_starts, speaker_ends in self.speaker_overview:
                print(str(speaker_id), end=" ")
            speaker_id = input("\nPlease enter the speaker ID of the speaker you want to remove, see folder 'faces_id' (if you don't want to remove a speaker, just press enter): ")
            
            # Error handling, if input is not one of the speaker IDs
            if speaker_id != "" and speaker_id not in [speaker_id for speaker_id, speaker_starts, speaker_ends in self.speaker_overview]:
                print("Please enter a valid speaker ID!")
                continue
            
            if speaker_id == "":
                break
            else:
                # If he wants to remove the last speaker, throw an error
                if len(self.speaker_overview) == 2:
                    print("You need at least two people for one team!")
                    continue
                self.remove_speaker(speaker_id)
    
    def remove_speaker(self, speaker_id):
        for i in range(len(self.speaker_overview)):
            if self.speaker_overview[i][0] == speaker_id:
                del self.speaker_overview[i]
                break
    
    def get_block_length(self):
        block_length = []
        start_time = 0
        while start_time < self.length_video:
            end_time = min(start_time + self.unit_of_analysis, self.length_video)
            block_length.append(end_time - start_time)
            start_time += self.unit_of_analysis
        return block_length
    
    # Split the speaker overview into blocks (based on unit of analysis)
    def split_speaker_overview(self) -> list:
        
        block_duration = self.unit_of_analysis

        block_speaker_data = []
        start_time = 0
        while start_time < self.length_video:
            end_time = min(start_time + block_duration, self.length_video)
            block_data = []
            for speaker_id, speaker_starts, speaker_ends in self.speaker_overview:
                speaker_block_starts = []
                speaker_block_ends = []
                for i in range(len(speaker_starts)):
                    segment_start = max(start_time, speaker_starts[i])
                    segment_end = min(end_time, speaker_ends[i])
                    if segment_start < segment_end:
                        speaker_block_starts.append(segment_start)
                        speaker_block_ends.append(segment_end)
                    elif segment_start == segment_end and i < len(speaker_starts)-1 and speaker_ends[i] < speaker_starts[i+1]:
                        # Special case when segment is exactly at block boundary and there is a gap before the next segment
                        speaker_block_starts.append(segment_start)
                        speaker_block_ends.append(segment_end)
                        speaker_block_starts.append(segment_end)
                        speaker_block_ends.append(speaker_starts[i+1])
                if speaker_block_starts:
                    block_data.append([speaker_id, speaker_block_starts, speaker_block_ends])
            block_speaker_data.append(block_data)
            start_time += block_duration

            
        return block_speaker_data