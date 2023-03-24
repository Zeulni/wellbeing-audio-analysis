
import numpy as np
import os
import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt

from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso
from sklearn.decomposition import PCA

from src.audio.utils.constants import PERMA_MODEL_DIR
from src.audio.utils.constants import PERMA_MODEL_RESULTS_DIR

class FeatureReduction():
    def __init__(self) -> None:
        pass
    
    # Drop all columns that have at least one NaN value
    def remove_nan_columns(self, data, filename) -> pd:
        # Stats in long data: 4656 rows have at least one NaN value, 3105 rows have all NaN values
        
        print("Columns in total: ", len(data.columns))
        
        # Store all columns where any value is NaN
        columns_with_nan = []
        for col in data.columns:
            if data[col].isnull().any():
                columns_with_nan.append(col)
                
        print("Columns with NaN values: ", len(columns_with_nan))
        
        # Remove the all the columns with NaN values
        data = data.drop(columns_with_nan, axis=1)
        
        # Overwrite existing csv with updated dataframe
        data.to_csv(os.path.join(PERMA_MODEL_DIR, filename + ".csv"))
        
        return data
    
    # Rule of thumb: 1 feature per 10 samples -> 8-9 features have to be selected
    def select_features_regression(self, data_X, data_y, database) -> pd:

        # TODO: values for normalized features
        if database == "short_data":
            alpha = 0.025
        elif database == "long_data":
            alpha = 0.05
        
        # TODO: values for standardized features
        # if database == "short_data":
        #     alpha = 0.1
        # elif database == "long_data":
        #     alpha = 0.2

        # Use MultiOutputRegressor to select the features for all target variables at once (0.1 original alpha value)
        selector = SelectFromModel(Lasso(alpha=alpha, max_iter=10000))
        
        # fit the feature selector
        selector.fit(data_X, data_y)
        
        # transform the feature matrix
        data_X_new = selector.transform(data_X)
        
        # convert the transformed feature matrix to a DataFrame
        data_X_new = pd.DataFrame(data_X_new, columns=data_X.columns[selector.get_support()])
        
        # print the selected features
        print(f"Amount selected features: {len(data_X_new.columns)}")
        
        feature_importance_dict = self.get_feature_importances(selector, data_X, data_y)
        
        # self.plot_feature_importance(selector, data_X, data_y)
    
        # Save the selected features as pickle file (to use it for inference later on)
        with open(os.path.join(PERMA_MODEL_RESULTS_DIR, database + "_selected_features.pkl"), "wb") as f:
            pkl.dump(data_X_new.columns, f)
        
        return data_X_new, feature_importance_dict
    
    def perform_pca(self, data_X, database) -> pd:
        
        # Rule of thumb: 1 feature per 10 samples -> 8-9 features have to be selected
        amount_features = 8
        pca = PCA(n_components=amount_features)
        pca.fit(data_X)
        
        # Save the PCA as pickle file (to use it for inference later on)
        with open(os.path.join(PERMA_MODEL_RESULTS_DIR, database + "_pca.pkl"), "wb") as f:
            pkl.dump(pca, f)
        
        reduced_data = pca.transform(data_X)
        
        print("Explained variance ratio with " + str(amount_features) + " features:" , round(sum(pca.explained_variance_ratio_),2))
        
        # Transform the reduced data back to a DataFrame
        reduced_data = pd.DataFrame(reduced_data, columns=[f"PC{i}" for i in range(1, amount_features + 1)])
        
        # self.interpret_pca_plot(pca, data_X)
        
        top_features = self.interpret_pca_list(pca, data_X)
        
        return reduced_data, top_features
    
    def interpret_pca_list(self, pca, data_X) -> dict:
        
        loadings = pca.components_
        
        feature_names = data_X.columns
        
        # Store the top 5 features that contribute the most to each principal component
        n_top_features = 5
        top_features = {}
        for i in range(loadings.shape[0]):
            pc_loadings = loadings[i, :]
            pc_top_features_idx = np.abs(pc_loadings).argsort()[::-1][:n_top_features]
            pc_top_features = [(feature_names[idx], pc_loadings[idx]) for idx in pc_top_features_idx]
            top_features['PC{}'.format(i+1)] = pc_top_features
            
        return top_features
    
    def interpret_pca_plot(self, pca, data_X) -> None:
        loadings = pca.components_
        
        feature_names = data_X.columns
        
        # Create a heatmap of the loadings
        fig, ax = plt.subplots()
        im = ax.imshow(loadings, cmap='coolwarm')

        # Add colorbar
        cbar = ax.figure.colorbar(im, ax=ax)

        # Set the x-axis tick labels to be the original feature names
        ax.set_xticks(np.arange(len(feature_names)))
        ax.set_xticklabels(feature_names)

        # Set the y-axis tick labels to be the principal component names
        ax.set_yticks(np.arange(loadings.shape[0]))
        ax.set_yticklabels(['PC{}'.format(i+1) for i in range(loadings.shape[0])])

        # Rotate the tick labels and set their alignment
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Loop over data dimensions and create text annotations
        for i in range(loadings.shape[0]):
            for j in range(loadings.shape[1]):
                text = ax.text(j, i, '{:.2f}'.format(loadings[i, j]),
                            ha="center", va="center", color="w")

        # Set the title
        ax.set_title("Which features contribute most to which PC")

        # Determine the size of the figure
        fig.set_size_inches(16, 9)

        # Show the plot
        plt.show()
    
    def plot_feature_importance(self, selector, data_X, data_y) -> None:
        
        # Plot the feature importance for every target variable (importances is a 5xN matrix)
        importances = selector.estimator_.coef_
        
        # Plot the feature importance for every target variable
        for i in range(5):
            importance_df = pd.DataFrame({'Feature': data_X.columns, 'Importance': importances[i]})
            
            # Sort the features by importance
            importance_df = importance_df.sort_values(by='Importance', ascending=False)
            
            importance_df.plot(kind='bar', x='Feature', y='Importance', title=f'Feature Importance for Target {data_y.columns[i]}')
            plt.show()
        
    def get_feature_importances(self, selector, data_X, data_y) -> dict:
        # Store for each target variable the most important features
        feature_importance_dict = {}
        
        # Plot the feature importance for every target variable (importances is a 5xN matrix)
        importances = selector.estimator_.coef_
        
        # Plot the feature importance for every target variable
        for i in range(5):
            importance_df = pd.DataFrame({'Feature': data_X.columns, 'Importance': importances[i]})
            
            # Sort the features by importance
            importance_df = importance_df.sort_values(by='Importance', ascending=False)
            
            # Store the most important features in a dictionary (features that have an importance != 0)
            # Every value is a list with feature names and the corresponding importance
            feature_importance_dict[data_y.columns[i]] = importance_df[importance_df['Importance'] != 0].values.tolist()
            
        return feature_importance_dict