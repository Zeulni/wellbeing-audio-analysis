import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

from src.audio.utils.constants import PERMA_MODEL_DIR
from src.audio.utils.constants import PERMA_MODEL_RESULTS_DIR

def write_results_to_csv(emotions_output, com_pattern_output, csv_path, video_name, asd_pipeline_tools) -> str:

    data_emotions_output = []
    for speaker_id, values in emotions_output.items():
        for key, val in values.items():
            for i, v in enumerate(val):
                col_name = f"{key}_{i+1}_{video_name}"
                data_emotions_output.append([speaker_id, col_name, v])

    df_emotions_output = pd.DataFrame(data_emotions_output, columns=["Speaker ID", "Emotion", "Value"])
    # Store the original order of the columns before the pivot
    original_emotions_order = df_emotions_output['Emotion'].unique()
    df_emotions_output = df_emotions_output.pivot(index="Speaker ID", columns="Emotion", values="Value")
    # Reorder the columns using the original order
    df_emotions_output = df_emotions_output.reindex(columns=original_emotions_order)
    
    data_com_pattern_output = []
    for speaker_id, values in com_pattern_output.items():
        for key, val in values.items():
            for i, v in enumerate(val):
                col_name = f"{key}_{i+1}_{video_name}"
                data_com_pattern_output.append([speaker_id, col_name, v])

    df_com_pattern_output = pd.DataFrame(data_com_pattern_output, columns=["Speaker ID", "ComPattern", "Value"])
    original_ComPattern_order = df_com_pattern_output['ComPattern'].unique()
    df_com_pattern_output = df_com_pattern_output.pivot(index="Speaker ID", columns="ComPattern", values="Value")  
    df_com_pattern_output = df_com_pattern_output.reindex(columns=original_ComPattern_order)
    
    
    df = df_emotions_output.join(df_com_pattern_output, on="Speaker ID")
    
    # print(df)
    
    # Also save the Speaker ID as a column
    df.reset_index(inplace=True)
    df.to_csv(csv_path, index=False)
    
    asd_pipeline_tools.write_to_terminal(f"CSV file saved to {csv_path}")
    
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
    # Drop the NaN values
    values = values[~np.isnan(values)]
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
    
def visualize_com_pattern(csv_path, unit_of_analysis, video_name, plotted_features) -> None:
    # Columns are the names of the columns to be shown in the plot
    
    # Read in the CSV file
    df = pd.read_csv(csv_path)
    
    # If columns is not a list with 3 elements (each a string), raise an error
    if not isinstance(plotted_features, list) or len(plotted_features) != 3:
        raise ValueError(f"plotted_features should be a list with 3 elements (each a string).")
    
    com_pattern_df = df[[col for col in df.columns if col.startswith(plotted_features[0]) or col.startswith(plotted_features[1]) or col.startswith(plotted_features[2]) or col == 'Speaker ID']]

    
    # Read in the CSV file
    df = pd.read_csv(csv_path)

    # Extract the corresponding columns
    col_1 = [col for col in com_pattern_df.columns if plotted_features[0] in col]
    col_2 = [col for col in com_pattern_df.columns if plotted_features[1] in col]
    col_3 = [col for col in com_pattern_df.columns if plotted_features[2] in col]

    # # Compute the range of the y-axis
    intermediate_df = df[[col for col in df.columns if col.startswith(plotted_features[0]) or col.startswith(plotted_features[1]) or col.startswith(plotted_features[2])]]
    values = intermediate_df.values.flatten()
    y_min = np.min(values)
    y_max = np.max(values)
    
    # In the list "plotted_features", replace "turns" with "utterances" and "overlaps" with "interruptions"
    # plotted_features = [col.replace("turns", "utterances") for col in plotted_features]
    # plotted_features = [col.replace("overlaps", "interruptions") for col in plotted_features]
    
    # Create subplots for each speaker
    fig, axs = plt.subplots(com_pattern_df.shape[0], 1, figsize=(10, 2*com_pattern_df.shape[0]))
    for i, speaker_id in enumerate(list(df["Speaker ID"])):
        
        col_1_data = com_pattern_df.loc[com_pattern_df["Speaker ID"] == speaker_id, col_1].values.tolist()[0]
        col_2_data = com_pattern_df.loc[com_pattern_df["Speaker ID"] == speaker_id, col_2].values.tolist()[0]
        col_3_data = com_pattern_df.loc[com_pattern_df["Speaker ID"] == speaker_id, col_3].values.tolist()[0]
        
        axs[i].plot(col_1_data, marker='o', label=plotted_features[0])
        axs[i].plot(col_2_data, marker='o', label=plotted_features[1])
        axs[i].plot(col_3_data, marker='o', label=plotted_features[2])
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
    
def read_final_database(folder) -> pd.DataFrame:
    # Read the csvs as dataframes (just run everytime again instead of checking if csv is available -> always up to date)
    data_folder = PERMA_MODEL_DIR / folder
    
    # Read all the csvs in the data folder as dataframes and append them in one dataframe
    data = pd.DataFrame()
    
    # Create a list of all csv files in data_folder and sort them
    csv_files = sorted([file for file in data_folder.glob("*.csv")])
    for file in csv_files:
        data = pd.concat([data, pd.read_csv(file)], axis=0)
        
    # Remove the rows "Unnamed: 0", "E-Mail-Adresse", "Alias", "First Name", "Last Name/Surname", "Day"
    data = data.drop(["Unnamed: 0", "E-Mail-Adresse", "Alias", "First Name", "Last Name/Surname", "Day"], axis=1)
    
    # Reset the index
    data = data.reset_index(drop=True)
    
    # * This is only done to correct for some feature names afterwards - in the future this does not have to be done
    # In every column header where "turns" is in the name, replace it with "utterances"
    data.columns = [col.replace("turns", "utterances") for col in data.columns]
    data.columns = [col.replace("overlaps", "interruptions") for col in data.columns]

    # Save the dataframe as csv
    data.to_csv(os.path.join(PERMA_MODEL_DIR, folder + ".csv"))
    
    return data

def create_dataset_for_sdm_plots(folder) -> None:
    # Read the csvs as dataframes (just run everytime again instead of checking if csv is available -> always up to date)
    data_folder = PERMA_MODEL_DIR / folder
    
    # Read all the csvs in the data folder as dataframes and append them in one dataframe
    data = pd.DataFrame()
    
    # Create a list of all csv files in data_folder and sort them
    csv_files = sorted([file for file in data_folder.glob("*.csv")])
    for file in csv_files:
        data = pd.concat([data, pd.read_csv(file)], axis=0)
        
    # * This is only done to correct for some feature names afterwards - in the future this does not have to be done
    # In every column header where "turns" is in the name, replace it with "utterances"
    data.columns = [col.replace("turns", "utterances") for col in data.columns]
    data.columns = [col.replace("overlaps", "interruptions") for col in data.columns]
    
    data = data.drop(["Unnamed: 0", "Alias", "Day", "P", "E", "R", "M", "A"], axis=1)
    
    # Remove the rows "Unnamed: 0", "E-Mail-Adresse", "Alias", "First Name", "Last Name/Surname", "Day"
    # data = data.drop(["Unnamed: 0", "E-Mail-Adresse", "Alias", "First Name", "Last Name/Surname", "Day"], axis=1)
    
    grouped = data.groupby(["E-Mail-Adresse", "First Name", "Last Name/Surname"])
    
    # Apply mean to all other columns and reset index
    new_df = grouped.mean().reset_index()
    
    # Now only keep the columns "arousal_std", "norm_num_interruptions_relative_median", "norm_speak_duration_relative_mean"
    new_df = new_df[["E-Mail-Adresse", "First Name", "Last Name/Surname", "arousal_std", "norm_num_interruptions_relative_median", "norm_speak_duration_relative_mean"]]

    # Save the dataframe as csv
    new_df.to_csv(os.path.join(PERMA_MODEL_DIR, "audio_sdm_data" + ".csv"))
    
    return data

def create_perma_results_folder() -> None:
    # Create a folder for the results of the PERMA analysis
    if not os.path.exists(PERMA_MODEL_RESULTS_DIR):
        os.mkdir(PERMA_MODEL_RESULTS_DIR)

def transform_test_data(data_X_test, data_y_test, nan_columns, normalize_scaler, gaussian_columns, gaussian_feature_scaler, non_gaussian_columns, nongaussian_feature_scaler, unique_features):
    
    # Step 1: Remove nan_columns from data_X_test
    data_X_test = data_X_test.drop(nan_columns, axis=1)
    
    # Step 2a: Normalize data_y_test using the normalize_scaler
    data_y_test_array = normalize_scaler.transform(data_y_test)
    data_y_test[["P", "E", "R", "M", "A"]] = data_y_test_array
    
    # Step 2b: Scale the gaussian features of data_X_Test using the gaussian_feature_scaler
    data_X_standardized = gaussian_feature_scaler.transform(data_X_test[gaussian_columns])
    data_X_standardized = pd.DataFrame(data_X_standardized, columns=gaussian_columns)
    
    # Step 2c: Scale the non-gaussian features of data_X_Test using the nongaussian_feature_scaler
    data_X_robust_scaled = nongaussian_feature_scaler.transform(data_X_test[non_gaussian_columns])
    data_X_robust_scaled = pd.DataFrame(data_X_robust_scaled, columns=non_gaussian_columns)
    
    data_X_test = pd.concat([data_X_standardized, data_X_robust_scaled], axis=1)
    
    # Step 3: Selecting the unique features
    data_X_test = data_X_test[unique_features]
    
    return data_X_test, data_y_test