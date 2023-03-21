import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle as pkl
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA

from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso


from src.audio.utils.constants import PERMA_MODEL_DIR

from src.audio.perma_model.perma_regressor import PermaRegressor

class PermaModel:
    def __init__(self) -> pd:
        pass
    
    def read_dataframe(self, folder) -> None:
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

        # Save the dataframe as csv
        data.to_csv(os.path.join(PERMA_MODEL_DIR, folder + ".csv"))
        
        return data
    
    # LOF works well for high dimensional data
    # See https://towardsdatascience.com/4-machine-learning-techniques-for-outlier-detection-in-python-21e9cfacb81d
    def remove_outliers(self, data_X, data_y) -> pd:
        
        # Extract the features
        X = data_X.values

        # Set LOF parameters for feature outlier detection
        n_neighbors = 20
        contamination = 0.03
        
        # Perform PCA for a more stable outlier detection (will just be used to find the indices, transformed data will not be used)
        # But it is not necessary to perform PCA for LOF, as no difference (at least not for my datasets)
        pca = PCA(n_components=9)
        pca.fit(X)
        X = pca.transform(X)

        # Fit the LOF model for feature outlier detection
        lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
        lof.fit(X)

        # Predict the outlier scores for feature outlier detection
        scores = lof.negative_outlier_factor_

        # Determine the threshold for outlier detection for feature outlier detection
        threshold = np.percentile(scores, 100 * contamination)

        # Identify the feature outliers
        outliers = X[scores < threshold]

        # Print the number of outliers and their indices for feature outlier detection
        print('Number of feature outliers:', len(outliers))
        outlier_rows = [i for i, x in enumerate(X) if any((x == y).all() for y in outliers)]

        # Create a new dataframe with outliers removed
        data_X_no_outliers = data_X.drop(index=outlier_rows)
        data_y_no_outliers = data_y.drop(index=outlier_rows)

        return data_X_no_outliers, data_y_no_outliers
    
    def plot_perma_pillars(self, data_y) -> None:

        # Plot each target variable as a box plot
        data_y.boxplot()
        plt.show()
    
    # Drop all columns that have at least one NaN value
    def handle_missing_values(self, data, filename) -> pd:
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

    def standardize_features(self, data_X, database) -> pd:
                
        scaler = StandardScaler()
        scaler.fit(data_X)
        
        # Save the scaler as pickle file (to use it for inference later on)        
        with open(os.path.join(PERMA_MODEL_DIR, database + "_scaler.pkl"), "wb") as f:
            pkl.dump(scaler, f)

        # fit and transform the DataFrame using the scaler
        array_standardized = scaler.transform(data_X)
        
        # Repalce the original columns with the standardized columns
        data_X_standardized = pd.DataFrame(array_standardized, columns=data_X.columns)
        
        return data_X_standardized
    
    # Rule of thumb: 1 feature per 10 samples -> 8-9 features have to be selected
    def select_features(self, data_X, data_y, database) -> pd:

        if database == "short_data":
            alpha = 0.1
        elif database == "long_data":
            alpha = 0.2

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
        with open(os.path.join(PERMA_MODEL_DIR, database + "_selected_features.pkl"), "wb") as f:
            pkl.dump(data_X_new.columns, f)
        
        return data_X_new, feature_importance_dict
    
    def perform_pca(self, data_X, database) -> pd:
        amount_features = 9
        pca = PCA(n_components=amount_features)
        pca.fit(data_X)
        
        # Save the PCA as pickle file (to use it for inference later on)
        with open(os.path.join(PERMA_MODEL_DIR, database + "_pca.pkl"), "wb") as f:
            pkl.dump(pca, f)
        
        reduced_data = pca.transform(data_X)
        
        print("Explained variance ratio with " + str(amount_features) + " features:" , round(sum(pca.explained_variance_ratio_),2))
        
        # Transform the reduced data back to a DataFrame
        reduced_data = pd.DataFrame(reduced_data, columns=[f"PC{i}" for i in range(1, amount_features + 1)])
        
        self.interpret_pca_plot(pca, data_X)
        
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

        
    def plot_pairplot(self, data_X, data_y, feature_importance_dict) -> None:
        
        sns.set(style="ticks", color_codes=True)
        
        # Create a scatter plot for each target variable (plot only the most important features against each target variable)
        
        for target, list_feature_importance in feature_importance_dict.items():
            
            # extract target variable and corresponding feature matrix
            y = data_y[target]
            # Only use the most important features for X
            X = data_X[[feature[0] for feature in list_feature_importance]]
            
            # concatenate target variable with feature matrix
            df = pd.concat([y, X], axis=1)
            
            # create pairplot for correlations of all features with the target variable
            sns.pairplot(df, height=1.4, aspect=0.8)
            plt.title(f"Correlations of most important features with {target}")
            plt.show()


    def plot_correlations(self, data_X, data_y, feature_importance_dict) -> None:
        
        # Plot correlations based on data_X, data_y and feature_importance_dict
        plt.figure(figsize=(16,10))
        
        # loop over each target variable
        for i, (target, list_feature_importance) in enumerate(feature_importance_dict.items()):
            
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

    
    def run(self): 

        # database_list = ["short_data", "long_data"]
        database_list = ["long_data"]
        
        for database in database_list:
            # Read the data
            data = self.read_dataframe(database)
            # Extract features and targets
            data_X = data.iloc[:, 5:] # features
            data_y = data.iloc[:, :5] # targets
            
            data_X = self.handle_missing_values(data_X, database)
            # TODO: create pipeline overview in Powerpoint (with number of features, and rows,...)
            # TODO: occams razor -> choose simpler model with e.g. depth 3 and 100 estimators
           
            # TODO: write in Overleaf the entire process (how calculated PERMA scores, how outlier, how scaled,...)
            # Only searches for outliers in X, outliers in PERMA (target) were already removed beforehand
            # Assumption: filter out ~3 outliers in the input data (data_X)
            data_X, data_y = self.remove_outliers(data_X, data_y) # Perform it before scaling features
            # y was already standardized
            data_X = self.standardize_features(data_X, database)
            data_X, feature_importance_dict = self.select_features(data_X, data_y, database)
            # TODO: Disadvantage PCA: it is hard to interpret the results
            data_X, top_features = self.perform_pca(data_X, database)
            # self.plot_pairplot(data_X, data_y, feature_importance_dict)
            # self.plot_perma_pillars(data_y)
            self.plot_correlations(data_X, data_y, feature_importance_dict)
            perma_regressor = PermaRegressor(data_X, data_y, database)
            perma_regressor.catboost_train()
            perma_regressor.xgboost_train()
            print("--------------------------------------------------")