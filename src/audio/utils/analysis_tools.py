import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from src.audio.utils.constants import VIDEOS_DIR

def write_results_to_csv(turn_taking_obj, speaking_duration_obj, overlaps_obj, video_name) -> str:
    # TODO: I only write the team features to a csv file (no individual features yet)
    # TODO: how to attach the results from one day into one csv file is open (e.g. pandas db as intermediate step? and first look if for that day one dataframe is already here? but what if run twice? and it just attaches to the old one)
    
    # Store the features in a pandas dataframe
    # The rows are the blocks and the columns are the features
    df = pd.DataFrame(columns=['block', 'number_turns_equality', 'speaking_duration_equality', 'norm_num_overlaps'])

    # Add the features to the dataframe
    df['block'] = turn_taking_obj.get("blocks_number_turns_equality")['block']
    df['number_turns_equality'] = turn_taking_obj.get("blocks_number_turns_equality")['number_turns_equality']
    df['speaking_duration_equality'] = speaking_duration_obj.get("blocks_speaking_duration_equality")['speaking_duration_equality']
    df['norm_num_overlaps'] = overlaps_obj.get("blocks_norm_num_overlaps")['norm_num_overlaps']
    
    # Store the pandas dataframe to a csv file
    filename = video_name + "_audio_analysis_results.csv"
    csv_path = str(VIDEOS_DIR / video_name / filename)
    df.to_csv(csv_path, index=False)
    
    return csv_path
    
def visualize_pattern(csv_path, unit_of_analysis, video_name) -> None:
    
    # Read the data from the csv file into a pandas dataframe
    df = pd.read_csv(csv_path)
    
    # Plot the data (one plot for each feature)
    fig, axes = plt.subplots(3, 1, figsize=(12, 9))
    
    # Set the amount of padding between subplots
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    
    # Add a heading to the plot
    fig.suptitle('Audio analysis results of ' + video_name)
    
    # Plot the first feature
    axes[0].plot(df['block'], df['number_turns_equality'])
    axes[0].set_title('Equality (based on number of turns) per block - 0 is perfectly equal')
    axes[0].set_ylabel('Equality')
    
    # Plot the second feature
    axes[1].plot(df['block'], df['speaking_duration_equality'])
    axes[1].set_title('Equality (based on speaking duration) per block - 0 is perfectly equal')
    axes[1].set_ylabel('Equality')
    
    # Plot the third feature
    axes[2].plot(df['block'], df['norm_num_overlaps'])
    axes[2].set_title('Norm. number of overlaps per block - per minute per speaker')
    axes[2].set_ylabel('Norm. number of overlaps')
    
    # Add a note to the bottom of the plot
    fig.text(0.5, 0.04, '1 unit on x-axis = ' + str(unit_of_analysis) + "s", ha='center')
    
    plt.show()
    
def visualize_individual_speaking_shares(speaking_duration):
    # Visualize the speaking_duration in a bar chart (one bar for each speaker)


    fig, ax = plt.subplots()
    ax.bar(speaking_duration["speaker"], speaking_duration["speaking_duration"])

    ax.set_xlabel("Speaker")
    ax.set_ylabel("Speaking Duration in Seconds")
    ax.set_title("Speaking Duration per Speaker")

    plt.show()
    
def visualize_emotions(emotions_output, unit_of_analysis, video_name):

    # Compute the range of the y-axis
    values = [value for data in emotions_output.values() for value in data.values()]
    y_min = np.min(values)
    y_max = np.max(values)

    # Create subplots for each speaker
    fig, axs = plt.subplots(len(emotions_output), 1, figsize=(10, 2*len(emotions_output)))
    for i, (speaker_id, data) in enumerate(emotions_output.items()):
        axs[i].plot(data['arousal'], marker='o', label='arousal')
        axs[i].plot(data['dominance'], marker='o', label='dominance')
        axs[i].plot(data['valence'], marker='o', label='valence')
        axs[i].set_title(f'Speaker {speaker_id}')
        axs[i].legend()
        axs[i].set_ylim(y_min, y_max)
        axs[i].set_xlim(1, len(data['arousal']))
        
    # Add a heading to the plot
    fig.suptitle('Audio analysis results of ' + video_name)

    # Set the x label for the bottom subplot
    axs[-1].set_xlabel('Unit of Analysis (1 unit = ' + str(unit_of_analysis) + 's)')

    # Adjust spacing between subplots
    plt.tight_layout()

    # Show the plot
    plt.show()