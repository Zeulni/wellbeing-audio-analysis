<h1 align="center">Toolkit for Audiovisual Speaker Diarization in Noisy Environments, Speech Feature Extraction, and Well-Being Prediction</h1>

*Repository for the master's thesis of Tobias Zeulner: Leveraging Speech Features for Automated Analysis of Well-Being in Teamwork Contexts*


<p align="center">
  <a href="https://www.python.org/downloads/release/python-380/"><img src="https://img.shields.io/badge/Python-3.8.10-blue" alt="Python 3.8.10" height="25"></a>
  <a href="https://github.com/Zeulni/wellbeing-audio-analysis/blob/main/LICENSE"><img src="https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=for-the-badge" alt="GitHub license" height="25"></a>
  <a href="https://www.linkedin.com/in/tobias-zeulner-893080169/"><img src="https://img.shields.io/badge/-LinkedIn-blue?style=flat&logo=linkedin" alt="Linkedin" height="25"></a>
</p>


<p align="center">
  <a href="#about-this-project">About this Project</a> •
  <a href="#key-features">Key Features</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#license">License</a>
</p>

## :book: About this Project

Current methods for assessing employee well-being rely primarily on irregular and time-consuming surveys, which limits proactive measures and support. This thesis addresses this problem by developing predictive algorithms to automatically assess well-being. The algorithms are based on audio data collected in teamwork contexts.
A dataset of 56 participants who worked in teams over a four-day period was curated. The positive emotion, engagement, relationships, meaning, and accomplishment (PERMA) framework consisting of five pillars ( developed by Seligman) was used to measure well-being. An audiovisual speaker diarization system was developed to enable the calculation of speech features at the individual level in a noisy environment. After extracting and selecting the most relevant features, regression, and classification algorithms were trained to predict well-being.

The best predictive model for each PERMA pillar is the two-class classification system. It achieves the following balanced accuracies: P: 78%, E: 50%, R: 74%, M: 61%, and A: 50%. 

The entire pipeline (see image below) and final models are provided in this GitHub repository.

## :star: Key Features

The four main building blocks of this toolbox are shown in the figure below.
[<img src="./docs/audio_wellbeing_analysis_overview.svg" alt="audio AI toolkit overview" />](./docs/audio_wellbeing_analysis_overview.svg)


0. Input Video:
    - mp4 or avi file
    - Stored in `src/audio/videos`
    - Filename provided in `configs/config.yaml`
    - Ideally 25 fps (otherwise processing takes longer)
1. Output of Audiovisual Speaker Diarization:
    - 1 folder with the same name as the video, containing all current and future results
    - 3 important files in this folder:
        1. RTTM file (“who spoke when”)
        2. Log file (for troubleshooting)
        3. “faces_id” folder, which contains all recognized speaker and their corresponding ID from the RTTM file
2. Output of Communication Pattern & Emotion Feature Calculation:
    - 1 csv file named "*VIDEONAME*_audio_analysis_results.csv" containing one row for each speaker with the corresponding features values over time as columns
3. Output of Feature Visualization:
    - 3 line charts for visualization of the feature values contained in the csv file
    - 3 features are plotted per chart (i.e., 9 time series in total)
4. Output of Well-Being Prediction:
    - 1 csv for the PERMA classification results (low/high well-being)
    - 1 csv for the PERMA regression results (continuous well-being scores either between 0-1 or 1-7)
    - 1 plot to visualize the regression results (also saved as “perma_spider_charts.png”)


The parts can be run separately if, for example, the prediction of well-being is not required but other downstream tasks such as the prediction of team performance are.

If you wish to exclude an individual from the analysis (e.g. either random person in the background or no informed consent), you can do so by:
1. performing only step 1 of the pipeline.
2. deleting the person's image in the `src/audio/videos/VIDEONAME/faces_id` folder.
3. perform the remaining steps of the pipeline (2,3,4). From now on, the corresponding person will be excluded from the analysis

If you want to change the name of a person from the ID to the real name, you can do it as follows:
1. perform only step 1 of the pipeline
2. rename the corresponding file name in the folder `src/audio/videos/VIDEONAME/faces_id` by adding two underscores after the ID followed by the name (e.g. change the name from "2.jpg" to "2__john.jpg")
3. execute the remaining steps of the pipeline (2,3,4). From now on, the analysis will use the real name, not the ID

## :gear: How To Use

1. I recommend using the same Python version as me to avoid conflicts (3.8.10). I also recommend to set up a new virtual environment using the venv module.

    <details>
    <summary>How to set up a virtual environment in Python 3 using the venv module (Windows)</summary>

    ```
    python -m venv venv
    .\venv\Scripts\activate
    ```
    </details>
    <details>
    <summary>How to set up a virtual environment in Python 3 using the venv module (MacOS/Linux)</summary>

    ```
    python3 -m venv venv
    source venv/bin/activate
    ```
    </details>

2. Then, install ffmpeg (which is needed to process the video recordings).
    <details>
    <summary>How to install ffmpeg on Windows/Linux/MacOS</summary>

    - [Windows 10](https://www.youtube.com/watch?v=r1AtmY-RMyQ&ab_channel=TroubleChute)
    - [Linux Ubuntu](https://www.youtube.com/watch?v=tf4p-SMw5jA&ab_channel=RickMakes)
    - [MacOS (M1)](https://www.youtube.com/watch?v=nmrjRqEIgGc&ab_channel=DavidHelmuth)

    </details>

3. Clone the repository and install the required packages:

    ```
    pip install -r requirements.txt
    ```

4. To process a video using this tool, follow the steps below (if you use it for the first time, you can leave the initial value in the configuration file (001) and go directly to the next step):

    1. Video Placement: Place the video you wish to process in the `src/audio/videos` directory. Ensure that the video file is in a format compatible with the project (mp4 or avi).
    2. Configuration File: Open the `configs/config.yaml` file. This file contains various parameters that control the processing of the video.
    3. Video Specification: In the configuration file, specify the filename of the video you placed in the `src/audio/videos` directory. Do not include the file extension in the filename. For instance, if your video file is called "my_video.mp4", you should enter "my_video".
    4. Parameter Adjustment: Review the other parameters in the configuration file. These parameters control various aspects of the video processing, and you may adjust them as necessary to suit your specific needs.

5. Run the main file:
    ```
    python main.py
    ```
    or 
    ```
    python3 main.py
    ```
    depending on your Python installation.

Note: Running the script on a GPU can accelerate it by a factor of 4x-8x.

Have fun! :sunglasses:

If you encounter any issues, please reach out to me or open a new issue.



## :page_facing_up: License

Distributed under the MIT License. See `LICENSE` for more information.

---

> Email:  <a href="mailto:tobias.zeulner@tum.de">tobias.zeulner@tum.de</a>
 &nbsp;&middot;&nbsp;
> LinkedIn: <a href="https://www.linkedin.com/in/tobias-zeulner-893080169/" target="_blank">Tobias Zeulner</a>

