import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from src.audio.utils.constants import VIDEOS_DIR

def write_results_to_csv(emotions_output, com_pattern_output, csv_path, video_name) -> str:

    data_emotions_output = []
    for speaker_id, values in emotions_output.items():
        for key, val in values.items():
            for i, v in enumerate(val):
                col_name = f"{key}_{i+1}_{video_name}"
                data_emotions_output.append([speaker_id, col_name, v])

    df_emotions_output = pd.DataFrame(data_emotions_output, columns=["Speaker ID", "Emotion", "Value"])
    df_emotions_output = df_emotions_output.pivot(index="Speaker ID", columns="Emotion", values="Value")
    
    data_com_pattern_output = []
    for speaker_id, values in com_pattern_output.items():
        for key, val in values.items():
            for i, v in enumerate(val):
                col_name = f"{key}_{i+1}_{video_name}"
                data_com_pattern_output.append([speaker_id, col_name, v])

    df_com_pattern_output = pd.DataFrame(data_com_pattern_output, columns=["Speaker ID", "ComPattern", "Value"])
    df_com_pattern_output = df_com_pattern_output.pivot(index="Speaker ID", columns="ComPattern", values="Value")  
    
    df = df_emotions_output.join(df_com_pattern_output, on="Speaker ID")
    
    print(df)
    
    # Also save the Speaker ID as a column
    df.reset_index(inplace=True)
    df.to_csv(csv_path, index=False)
    
    return
    
def visualize_individual_speaking_shares(speaking_duration):
    # Visualize the speaking_duration in a bar chart (one bar for each speaker)


    fig, ax = plt.subplots()
    ax.bar(speaking_duration["speaker"], speaking_duration["ind_speaking_share"])

    ax.set_xlabel("Speaker")
    ax.set_ylabel("Speaking Duration in Seconds")
    ax.set_title("Speaking Duration per Speaker")

    plt.show()
    
    
def visualize_emotions(csv_path, unit_of_analysis, video_name):
    
    # Read in the CSV file
    df = pd.read_csv(csv_path)
    
    # Only keep the emotion columns (if it starts with dominance,...) + the speaker ID
    emotions_df = df[[col for col in df.columns if col.startswith('dominance') or col.startswith('arousal') or col.startswith('valence') or col == 'Speaker ID']]

    # Extract the columns containing arousal, dominance, and valence
    arousal_cols = [col for col in emotions_df.columns if 'arousal' in col]
    dominance_cols = [col for col in emotions_df.columns if 'dominance' in col]
    valence_cols = [col for col in emotions_df.columns if 'valence' in col]

    # # Compute the range of the y-axis
    intermediate_df = df[[col for col in df.columns if col.startswith('dominance') or col.startswith('arousal') or col.startswith('valence')]]
    values = intermediate_df.values.flatten()
    y_min = np.min(values)
    y_max = np.max(values)

    # Create subplots for each speaker
    fig, axs = plt.subplots(emotions_df.shape[0], 1, figsize=(10, 2*emotions_df.shape[0]))
    for i, speaker_id in enumerate(list(df["Speaker ID"])):
        
        arousal_data = emotions_df.loc[emotions_df["Speaker ID"] == speaker_id, arousal_cols].values.tolist()[0]
        dominance_data = emotions_df.loc[emotions_df["Speaker ID"] == speaker_id, dominance_cols].values.tolist()[0]
        valence_data = emotions_df.loc[emotions_df["Speaker ID"] == speaker_id, valence_cols].values.tolist()[0]       
        
        axs[i].plot(arousal_data, marker='o', label='arousal')
        axs[i].plot(dominance_data, marker='o', label='dominance')
        axs[i].plot(valence_data, marker='o', label='valence')
        axs[i].set_title(f'Speaker {speaker_id}')
        axs[i].legend()
        axs[i].set_ylim(y_min, y_max)
        
    # Add a heading to the plot
    fig.suptitle('Audio analysis results of ' + video_name)

    # Set the x label for the bottom subplot
    axs[-1].set_xlabel('Unit of Analysis (1 unit = ' + str(unit_of_analysis) + 's)')

    # Adjust spacing between subplots
    plt.tight_layout()

    # Show the plot
    plt.show()
    
def visualize_com_pattern(csv_path, unit_of_analysis, video_name, columns) -> None:
    # Columns are the names of the columns to be shown in the plot
    
    # If columns is not a list with 3 elements (each a string), raise an error
    if not isinstance(columns, list) or len(columns) != 3:
        raise ValueError(f"Columns should be a list with 3 elements (each a string).")
    
    
    

    emotions_df = df[[col for col in df.columns if col.startswith(columns[0]) or col.startswith('norm_speak_duration_relative') or col.startswith('norm_num_overlaps_relative') or col == 'Speaker ID']]

    
    # Read in the CSV file
    df = pd.read_csv(csv_path)

    # Extract the corresponding columns
    number_turns_cols = [col for col in emotions_df.columns if 'ind_number_turns_share_team' in col]
    speaking_duration_cols = [col for col in emotions_df.columns if 'ind_speaking_share_team' in col]

    # # Compute the range of the y-axis
    intermediate_df = df[[col for col in df.columns if col.startswith('ind_speaking_share_team') or col.startswith('ind_number_turns_share_team')]]
    values = intermediate_df.values.flatten()
    y_min = np.min(values)
    y_max = np.max(values)

    # Create subplots for each speaker
    fig, axs = plt.subplots(emotions_df.shape[0], 1, figsize=(10, 2*emotions_df.shape[0]))
    for i, speaker_id in enumerate(list(df["Speaker ID"])):
        
        number_turns_data = emotions_df.loc[emotions_df["Speaker ID"] == speaker_id, number_turns_cols].values.tolist()[0]
        speaking_duration_data = emotions_df.loc[emotions_df["Speaker ID"] == speaker_id, speaking_duration_cols].values.tolist()[0]
        
        axs[i].plot(number_turns_data, marker='o', label='ind. number turns shares (team)')
        axs[i].plot(speaking_duration_data, marker='o', label='ind. speaking shares (team)')
        axs[i].set_title(f'Speaker {speaker_id}')
        axs[i].legend()
        axs[i].set_ylim(y_min, y_max)
        
    # Add a heading to the plot
    fig.suptitle('Audio analysis results of ' + video_name)

    # Set the x label for the bottom subplot
    axs[-1].set_xlabel('Unit of Analysis (1 unit = ' + str(unit_of_analysis) + 's)')

    # Adjust spacing between subplots
    plt.tight_layout()

    # Show the plot
    plt.show()