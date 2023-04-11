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
            
            # G: arousal_min, n-G: norm_num_overlaps_absolute_min
            
            # First, plot only the distributions of arousal_min
            # Plot the histogram of col2
            # data_X['norm_num_overlaps_absolute_min'].plot(kind='hist', edgecolor='black')

            # # Add labels and title
            # plt.xlabel('Values')
            # plt.ylabel('Frequency')
            # plt.title('Distribution of the norm_num_overlaps_absolute_min Feature')
            
            # # Save the figure
            # plt.savefig(PERMA_MODEL_RESULTS_DIR / "norm_num_overlaps_absolute_min_distribution.png", dpi=600)

            # plt.show()
            
            
            
            for feature_name in FEATURE_NAMES:
                feature_cols = [col for col in data_X.columns if feature_name in col]
                data_X_feature = data_X[feature_cols]
                
                # Plot for each data_X_feature the correlation matrix (heatmap)
                sns.pairplot(data_X_feature)
                plt.tight_layout()
                plt.show()
    
    def plot_perma_pillars(self, data_y) -> None:

        labels = ["P", "E", "R", "M", "A"]

        # Plot each target variable as a box plot, using the labels
        data_y.boxplot(labels=labels, patch_artist=True)

        # Set the title
        plt.title("Distribution of each PERMA Pillar")
        plt.ylabel("PERMA Score")

        # Increase font size of x-axis labels
        plt.xticks(fontsize=14)

        # Remove the grid
        plt.grid(False)

        # Save the figure
        plt.savefig(PERMA_MODEL_RESULTS_DIR / "perma_scores.png", dpi=600)
                
        plt.show()
        
    def plot_correlations_with_target(self, data_X, data_y) -> None:
        
        # Plot correlations based on data_X, data_y and feature_importance_dict
        plt.figure(figsize=(16,10))
        
        target_values = ["P", "E", "R", "M", "A"]
        
        # loop over each target variable
        for i, target in enumerate(target_values):
            
            # extract target variable and corresponding feature matrix
            y = data_y[target]
            # Only use the most important features for X
            # X = data_X[[feature[0] for feature in list_feature_importance]]
            
            # Plot with all features (not just the important ones), as all features are used for the model later on
            X = data_X
            
            # create a correlation matrix for the features and target variable
            corr_matrix = X.corrwith(y)
            
            # Sort the correlations from highest to lowest
            corr_matrix = corr_matrix.sort_values(ascending=False)
            
            # plot the heatmap
            ax = plt.subplot(2, 3, i+1)
            sns.heatmap(corr_matrix.to_frame(), annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=ax)
            plt.title(f"Correlation with Target {target}")
            plt.tight_layout()
            
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