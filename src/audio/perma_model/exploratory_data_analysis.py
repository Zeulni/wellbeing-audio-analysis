import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from src.audio.utils.constants import FEATURE_NAMES
from src.audio.utils.constants import PERMA_MODEL_RESULTS_DIR

class ExploratoryDataAnalysis():
    def __init__(self):
        pass
    
    def plot_correlation_matrices_feature_names(self, data_X, data_y) -> None:

        for feature_name in FEATURE_NAMES:
            feature_cols = [col for col in data_X.columns if feature_name in col]
            data_X_feature = data_X[feature_cols]
            
            # Plot for each data_X_feature the correlation matrix (heatmap)
            corr = data_X_feature.corr()
            sns.heatmap(corr, annot=True, cmap="YlGnBu")
            plt.tight_layout()
            
            # Save the plot to a file (in folder stored in PERMA_MODEL_RESULTS_DIR)
            plt.savefig(PERMA_MODEL_RESULTS_DIR / f"correlation_matrix_{feature_name}.png", dpi=600)
            
            plt.show()

            
    def plot_correlation_matrices_feature_types(self, data_X, data_y) -> None:

        features_types = ["_mean", "_median", "_q25", "_q75", "_std", "_var", "_min", "_max", "_slope"]

        for feature_type in features_types:
            feature_cols = [col for col in data_X.columns if feature_type in col]
            data_X_feature = data_X[feature_cols]
            
            # Plot for each data_X_feature the correlation matrix (heatmap)
            corr = data_X_feature.corr()
            sns.heatmap(corr, annot=True, cmap="YlGnBu")
            plt.tight_layout()
            plt.show()
            
    def plot_pairplots_feature_names(self, data_X, data_y) -> None:
            
            # G: arousal_min, n-G: norm_num_interruptions_absolute_min
            
            # First, plot only the distributions of arousal_min
            # Plot the histogram of col2
            data_X['arousal_min'].plot(kind='hist', edgecolor='black', color='#666666')

            # Add labels and title
            plt.xlabel('arousal_min', fontsize=14)
            plt.ylabel('Frequency', fontsize=14)
            # plt.title('Distribution of the arousal_min Feature', fontsize=12)
            
            # Remove the grid
            plt.grid(True, alpha=0.5)
            
            # Save the figure
            plt.savefig(PERMA_MODEL_RESULTS_DIR / "arousal_min_distribution.png", dpi=600)

            plt.show()
            
            
            
            for feature_name in FEATURE_NAMES:
                feature_cols = [col for col in data_X.columns if feature_name in col]
                data_X_feature = data_X[feature_cols]
                
                # Plot for each data_X_feature the correlation matrix (heatmap)
                sns.pairplot(data_X_feature)
                plt.tight_layout()
                plt.show()
    
    def plot_perma_pillars(self, data_y) -> None:

        labels = ["P", "E", "R", "M", "A"]

        # Define properties for the median line
        medianprops = dict(linestyle='-', linewidth=1, color='#000000')

        # Plot each target variable as a box plot, using the labels
        data_y.boxplot(labels=labels, patch_artist=True, color='#666666', medianprops=medianprops)

        # Set the title
        # plt.title("Distribution of each PERMA Pillar")
        plt.ylabel("Score", fontsize=14)
        plt.xlabel("PERMA pillars", fontsize=14)

        # Increase font size of x-axis labels
        plt.xticks(fontsize=14)

        # Remove the grid
        plt.grid(True, alpha=0.5)

        # Save the figure
        plt.savefig(PERMA_MODEL_RESULTS_DIR / "perma_scores_distribution.png", dpi=600)
                
        plt.show()
        
    def plot_correlations_with_target(self, data_X, data_y, perma_feature_list) -> None:
        
        # Plot correlations based on data_X, data_y and feature_importance_dict
        plt.figure(figsize=(16,6))
        
        target_values = ["P", "E", "R", "M", "A"]
        
        # loop over each target variable
        for i, target in enumerate(target_values):
            
            # extract target variable and corresponding feature matrix
            y = data_y[target]
            # Only use the most important features for X
            # X = data_X[[feature[0] for feature in list_feature_importance]]
            
            X = data_X[perma_feature_list[i]]
            
            # Plot with all features (not just the important ones), as all features are used for the model later on
            # X = data_X
            
            # create a correlation matrix for the features and target variable
            corr_matrix = X.corrwith(y)
            
            # Sort the correlations from highest to lowest
            corr_matrix = corr_matrix.sort_values(ascending=False)
            
            # plot the heatmap
            ax = plt.subplot(2, 3, i+1)
            
            sns.heatmap(corr_matrix.to_frame(), annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=ax, yticklabels=True)
            # sns.set(font_scale=1.5)
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
            ax.set_xticklabels([])        
            # Adjust tick parameters to ensure that ticks are visible
            ax.tick_params(axis='both', which='both', length=0, width=0, pad=3)
            ax.tick_params(axis='y', which='major', length=6, width=1)
            # ax.set_yticklabels([])  
            # plt.title(f"Correlation with Pillar {target}")
            # Have a bold title
            plt.title(f"Correlation with Pillar {target}", fontweight="bold", pad=10)
            # plt.tight_layout()
            # Add some space between the subplots
            plt.subplots_adjust(left=0.21, right=0.96, top=0.9, wspace=2.5, hspace=0.3)
        
        # Save the plot to a file (in folder stored in PERMA_MODEL_RESULTS_DIR)
        plt.savefig(PERMA_MODEL_RESULTS_DIR / "selected_feature_perma_correlations.png", dpi=900)
        plt.show()
        
    def plot_pairplot_final_features(self, data_X, data_y, feature_importance_dict) -> None:
        
        sns.set(style="ticks", color_codes=True)
        
        # Create a scatter plot for each target variable (plot only the most important features against each target variable)
        
        for target, list_feature_importance in feature_importance_dict.items():
            
            # extract target variable and corresponding feature matrix
            y = data_y[target]
            # Only use the most important features for X
            # X = data_X[[feature[0] for feature in list_feature_importance]]
            X = data_X
            
            # concatenate target variable with feature matrix
            df = pd.concat([y, X], axis=1)
            
            # create pairplot for correlations of all features with the target variable
            sns.pairplot(df, height=1.4, aspect=0.8)
            plt.title(f"Correlations of most important features with {target}")
            plt.show()
            
    def print_stats(self, data_X) -> None:
        
        # Set the display options to show all columns without truncation
        pd.set_option('display.max_columns', None)
        # pd.set_option('display.max_rows', None)
        # pd.set_option('display.width', None)
        # pd.set_option('display.max_colwidth', -1)

        # Use the info method to get a summary of the data
        print(data_X.describe())