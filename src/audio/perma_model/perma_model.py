import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle as pkl
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor

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
    def detect_outliers(self, data_X, data_y) -> pd:
        
        # Extract the features and target variables
        X = data_X.values # features
        y = data_y.values # targets

        # Set LOF parameters for feature outlier detection
        n_neighbors_f = 20
        contamination_f = 0.05

        # Set LOF parameters for target outlier detection
        n_neighbors_t = 20
        contamination_t = 0.05

        # Fit the LOF model for feature outlier detection
        lof_f = LocalOutlierFactor(n_neighbors=n_neighbors_f, contamination=contamination_f)
        lof_f.fit(X)

        # Predict the outlier scores for feature outlier detection
        scores_f = lof_f.negative_outlier_factor_

        # Determine the threshold for outlier detection for feature outlier detection
        threshold_f = np.percentile(scores_f, 100 * contamination_f)

        # Identify the feature outliers
        outliers_f = X[scores_f < threshold_f]

        # Fit the LOF models for target outlier detection
        lof_t = []
        for i in range(y.shape[1]):
            lof = LocalOutlierFactor(n_neighbors=n_neighbors_t, contamination=contamination_t)
            lof.fit(y[:, i].reshape(-1, 1))
            lof_t.append(lof)

        # Predict the outlier scores for target outlier detection
        scores_t = np.zeros_like(y)
        for i in range(y.shape[1]):
            scores_t[:, i] = lof_t[i].negative_outlier_factor_.reshape(-1)

        # Determine the threshold for outlier detection for target outlier detection
        threshold_t = np.percentile(scores_t, 100 * contamination_t, axis=0)

        # Identify the target outliers
        # TODO: shape is different than outliers_f -> cause for problem with printing?
        outliers_t = y[scores_t < threshold_t]

        # Print the number of outliers and their indices for feature outlier detection
        print('Number of feature outliers:', len(outliers_f))
        print('Feature outlier indices:', [i for i, x in enumerate(X) if any((x == y).all() for y in outliers_f)])

        # Print the number of outliers and their indices for target outlier detection
        print('Number of target outliers:', len(outliers_t))
        print('Target outlier indices:', [i for i, x in enumerate(y) if any((x == y).all() for y in outliers_t)])
    
    def plot_perma_pillars(self, data_y) -> None:
        
        # # Plot each target variable as a scatter plot
        # for col in data_y.columns:
        #     plt.scatter(data_y.index, data_y[col], label=col)

        # plt.legend()
        # plt.show()

        # Plot each target variable as a box plot
        data_y.boxplot()
        plt.show()
    
    def handle_missing_values(self, data, filename) -> pd:
        
        # TODO: any or all?
        # Stats in long data: 4656 rows have at least one NaN value, 3105 rows have all NaN values
        
        # Store all columns where all values are NaN
        columns_with_nan = []
        for col in data.columns:
            if data[col].isnull().any():
                columns_with_nan.append(col)
                
        # Print the columns with NaN values
        print("Columns with NaN values: ", len(columns_with_nan))
        
        # TODO: open: how to handle missing values? Drop them? Impute them?
        # TODO: store deleted columns, needed for inference
        # Remove the columns with all NaN values
        data = data.drop(columns_with_nan, axis=1)
        
        # Store the columns with NaN values in a pickle file
        with open(os.path.join(PERMA_MODEL_DIR, filename + "_columns_with_nan.pkl"), "wb") as f:
            pkl.dump(columns_with_nan, f)
        
        # Overwrite existing csv with updated dataframe
        data.to_csv(os.path.join(PERMA_MODEL_DIR, filename + ".csv"))
        
        return data

    def standardize_features(self, data_X) -> pd:
        
        scaler = StandardScaler()
        scaler.fit(data_X)
        
        # Save the scaler as pickle file (to use it for inference later on)        
        with open(os.path.join(PERMA_MODEL_DIR, "data_short_scaler.pkl"), "wb") as f:
            pkl.dump(scaler, f)

        # fit and transform the DataFrame using the scaler
        array_standardized = scaler.transform(data_X)
        
        # Repalce the original columns with the standardized columns
        data_X_standardized = pd.DataFrame(array_standardized, columns=data_X.columns)
        
        return data_X_standardized
    
    def select_features(self, data_X, data_y) -> pd:

        # define the feature selector (Lasso for regularization to avoid overfitting with high dimensional data and few samples)
        selector = SelectFromModel(Lasso(alpha=0.1, max_iter=10000))
        
        data_y_X_dict = {}

        # loop over each target variable
        for target in range(5):
            # extract the target variable
            y = data_y.iloc[:, target]

            # fit the feature selector
            selector.fit(data_X, y)

            # transform the feature matrix
            data_X_new = selector.transform(data_X)
            
            # convert the transformed feature matrix to a DataFrame
            data_X_new = pd.DataFrame(data_X_new, columns=data_X.columns[selector.get_support()])
            
            # Store for each data_y.columns[target] the y and data_X_new in a dictionary
            data_y_X_dict[data_y.columns[target]] = [y, data_X_new]

            # print the selected features
            print(f"Selected features for target {data_y.columns[target]}: {data_X_new.columns}")
                        
            # self.plot_feature_importance(selector, data_X, target)
    
        # TODO: save the names of the selected features in a pickle file (to use it for inference later on)
        
        return data_y_X_dict
    
    def plot_feature_importance(self, selector, data_X, target) -> None:
        # get the feature importances
        importances = selector.estimator_.coef_
        # plot the feature importances
        importance_df = pd.DataFrame({'Feature': data_X.columns, 'Importance': importances})
        
        # Sort the features by importance
        importance_df = importance_df.sort_values(by='Importance', ascending=False)
        
        importance_df.plot(kind='bar', x='Feature', y='Importance', title=f'Feature Importance for Target {target}')
        plt.show()
        
    def plot_pairplot(self, data_y_X_dict) -> None:
        
        sns.set(style="ticks", color_codes=True)
        
        # Create a scatter plot for each target variable (plot all features against each target variable)
        # Create a matrix of scatter plots
        for target, data in data_y_X_dict.items():
            # extract target variable and corresponding feature matrix
            y = data[0]
            X = data[1]

            # concatenate target variable with feature matrix
            df = pd.concat([y, pd.DataFrame(X)], axis=1)

            # create pairplot for correlations of all features with the target variable
            sns.pairplot(df, height=1.4, aspect=0.8)
            plt.title(f"Correlations of all features with {target}")
            plt.show()

    def plot_correlations(self, data_y_X_dict) -> None:

        # define the figure size
        plt.figure(figsize=(16,10))

        # loop over each target variable
        for i, (target, (y, X)) in enumerate(data_y_X_dict.items()):
            
            # create a correlation matrix for the features and target variable
            corr_matrix = X.corrwith(y)
            
            # plot the heatmap
            ax = plt.subplot(2, 3, i+1)
            sns.heatmap(corr_matrix.to_frame(), annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=ax)
            plt.title(f"Correlation with Target {target}")
            plt.tight_layout()
            
        plt.show()

    
    def run(self): 

        short_data = self.read_dataframe("short_data")
        short_data_X = short_data.iloc[:, 5:] # features
        short_data_y = short_data.iloc[:, :5] # targets
        
        # long_data = self.read_dataframe("long_data")
        # long_data_X = long_data.iloc[:, 5:] # features
        # long_data_y = long_data.iloc[:, :5] # targets
        
        # TODO: open: how exactly to perform outlier detection? Only on PERMA scores? how to handle the outliers?
        # self.detect_outliers(short_data_X, short_data_y)
    
        # self.plot_perma_pillars(short_data_y)
        
        short_data = self.handle_missing_values(short_data, "short_data")
        # long_data = self.handle_missing_values(long_data, "long_data")
        
        short_data_X = self.standardize_features(short_data_X)
        
        data_y_X_dict = self.select_features(short_data_X, short_data_y)
        
        # self.plot_pairplot(data_y_X_dict)
        
        # self.plot_correlations(data_y_X_dict)
        
        perma_regressor = PermaRegressor(data_y_X_dict)
        perma_regressor.catboost_train()
        
        print("test")