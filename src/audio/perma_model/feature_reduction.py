
import numpy as np
import os
import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_selection import SelectFromModel, SelectKBest, mutual_info_regression
from sklearn.linear_model import Lasso
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import RFECV
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler

from src.audio.utils.constants import PERMA_MODEL_DIR
from src.audio.utils.constants import PERMA_MODEL_RESULTS_DIR

class FeatureReduction():
    def __init__(self) -> None:
        pass
    
    # Drop all columns that have at least one NaN value
    def remove_nan_columns(self, data, filename) -> pd:
        # Stats in long data: 4656 rows have at least one NaN value, 3105 rows have all NaN values
        
        print("Columns in total: ", len(data.columns))
        print("Rows in total: ", len(data))
        
        # Store all columns where any value is NaN
        columns_with_nan = []
        for col in data.columns:
            if data[col].isnull().any():
                columns_with_nan.append(col)
        
        print("Number of features after removing NaN columns: ", len(data.columns) - len(columns_with_nan))
        
        # Remove the all the columns with NaN values
        data = data.drop(columns_with_nan, axis=1)
        
        # Overwrite existing csv with updated dataframe
        data.to_csv(os.path.join(PERMA_MODEL_DIR, filename + ".csv"))
        
        return data
    
    def finding_best_k_mutual_info(self, data_X, data_y) -> pd:
        
        # TODO: also test for other targets
        data_y = data_y["P"]
        
        # Define the pipeline with feature selection and regression
        selector = SelectKBest(mutual_info_regression)

        # Define the grid search parameters
        # param_grid = {'selector__k': np.arange(1, data_X.shape[1] + 1)}
        # in 500 steps
        param_grid = {'k': np.arange(1, data_X.shape[1] + 1, 500)}

        # Define the grid search object with the multi-output regressor and the parameter grid
        grid_search = GridSearchCV(selector, param_grid=param_grid, scoring='neg_mean_absolute_error')

        # Fit the grid search object on the data
        grid_search.fit(data_X, data_y)

        # Extract the results of the grid search
        results = grid_search.cv_results_

        # Plot the mean absolute error vs. k
        plt.plot(results['param_k'], -1 * results['mean_test_score'])
        plt.xlabel('Number of Features')
        plt.ylabel('Mean Absolute Error')
        plt.title('Mean Absolute Error vs. Number of Features')
        plt.show()
    
    def select_features_mutual_info(self, data_X, data_y, database, k_per_target) -> pd:
            
        k = k_per_target
            
        # Create an empty DataFrame to store the selected features for each target variable
        data_X_new = pd.DataFrame()

        # Loop over each target variable
        for i, col in enumerate(data_y.columns):
            
            # Select the target variable and the corresponding features from the input data
            data_y_i = data_y.iloc[:, i]
            data_X_i = data_X

            # Create a pipeline with feature selection and regression
            selector = SelectKBest(mutual_info_regression, k=k)

            # Fit the pipeline to the current target variable
            selector.fit(data_X_i, data_y_i)

            # Transform the feature matrix
            data_X_i_new = selector.transform(data_X_i)

            # Convert the transformed feature matrix to a DataFrame
            data_X_i_new = pd.DataFrame(data_X_i_new, columns=data_X_i.columns[selector.get_support()])

            # Append the selected features for the current target variable to the result DataFrame
            data_X_new = pd.concat([data_X_new, data_X_i_new], axis=1)

        # If column names are not unique, delete the duplicates
        data_X_new = data_X_new.loc[:, ~data_X_new.columns.duplicated()]

        # Print the selected features
        print(f"Amount selected features: {len(data_X_new.columns)}")

        # Save the selected features as a pickle file (to use it for inference later on)
        with open(os.path.join(PERMA_MODEL_RESULTS_DIR, database + "_selected_features_mutual.pkl"), "wb") as f:
            pkl.dump(data_X_new.columns, f)

        return data_X_new
    
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
        
        # feature_importance_dict = self.get_feature_importances(selector, data_X, data_y)
        
        # self.plot_feature_importance(selector, data_X, data_y)
    
        # Save the selected features as pickle file (to use it for inference later on)
        with open(os.path.join(PERMA_MODEL_RESULTS_DIR, database + "_selected_features_regression.pkl"), "wb") as f:
            pkl.dump(data_X_new.columns, f)
        
        return data_X_new
    
    def perform_pca(self, data_X, database, amount_features) -> pd:
        
        # Rule of thumb: 1 feature per 10 samples -> 8-9 features have to be selected
        pca = PCA()
        pca.fit(data_X)
        
        # Save the PCA as pickle file (to use it for inference later on)
        with open(os.path.join(PERMA_MODEL_RESULTS_DIR, database + "_pca.pkl"), "wb") as f:
            pkl.dump(pca, f)
        
        reduced_data = pca.transform(data_X)[:, :amount_features]
        
        # print("Explained variance ratio with " + str(amount_features) + " features:" , round(sum(pca.explained_variance_ratio_),2))
        print("Explained variance ratio with " + str(amount_features) + " features:" , round(sum(pca.explained_variance_ratio_[:amount_features]),2))
        
        # Transform the reduced data back to a DataFrame
        reduced_data = pd.DataFrame(reduced_data, columns=[f"PC{i}" for i in range(1, amount_features + 1)])
        
        self.print_pareto_plot(pca, amount_features)
        # self.interpret_pca_plot(pca, data_X)
        
        # top_features = self.interpret_pca_list(pca, data_X)
        
        return reduced_data
    
    def print_pareto_plot(self, pca, amount_features) -> None:
        # Calculate the cumulative sum of explained variances
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

        # Create a Pareto chart
        fig, ax = plt.subplots()
        ax.bar(range(len(pca.explained_variance_ratio_)), pca.explained_variance_ratio_, alpha=0.5, align='center', label='Individual explained variance')
        ax.plot(range(len(pca.explained_variance_ratio_)), cumulative_variance, '-o', label='Cumulative explained variance')
        ax.set_xticks(range(len(pca.explained_variance_ratio_)))
        ax.set_xticklabels(['PC{}'.format(i) for i in range(1,len(pca.explained_variance_ratio_)+1)])
        ax.set_xlabel('Principal components')
        ax.set_ylabel('Explained variance ratio')
        ax.legend(loc='best')
        plt.show()
    
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
    
    # Reduce features with low variance
    def variance_thresholding(self, data_X, best_param) -> pd.DataFrame:
        
        
        # Scaling from before has no influence (same as directly normalized)
        scaler = MinMaxScaler()
        data_X_normalized = scaler.fit_transform(data_X)
        
        # variances = np.var(data_X_normalized, axis=0)   
        
        # Debugging
        variances = pd.DataFrame({'Feature': data_X.columns, 'Variance': np.var(data_X_normalized, axis=0)})
        
        threshold = best_param['threshold_variance']
        
        # Create a VarianceThreshold feature selector
        selector = VarianceThreshold(threshold=threshold)
        
        # Fit the selector to the data (data is already normalized)
        selector.fit(data_X_normalized)
        
        # Get the indices of the features that are kept
        kept_indices = selector.get_support(indices=True)
        
        print(f'Number of features after variance thresholding: {len(kept_indices)}')
        
        # Create a new DataFrame with only the kept features
        reduced_data_X = data_X.iloc[:, kept_indices]
        
        return reduced_data_X
    
    # Identify highly correlated features to drop them (1 feature will stay, but it removes duplicates)
    def correlation_thresholding(self, data_X, best_param) -> pd.DataFrame:
        
        # Plot the correlation matrix as a heatmap
        plt.figure(figsize=(16, 9))
        sns.heatmap(data_X.corr(), annot=True)
        plt.show()
        
        threshold = best_param['threshold_correlation']
        
        # Compute correlation matrix with absolute values
        matrix = data_X.corr().abs()

        # Create a boolean mask
        mask = np.triu(np.ones_like(matrix, dtype=bool))

        # Subset the matrix
        reduced_matrix = matrix.mask(mask)
        
        # Dict that stores for each feature the features that are highly correlated
        correlated_features = []

        # Find cols that meet the threshold
        to_drop = set()
        for i, col in enumerate(reduced_matrix):
            # Check if any other column has correlation coefficient greater than threshold
            if any(reduced_matrix[col] > threshold):
                # Add all other columns with high correlation to the set of columns to drop
                for c in reduced_matrix.columns:
                    value = reduced_matrix[col][c]
                    if c != col and value > threshold:
                        to_drop.update([c])
                        # Add the correlated feature to the list (at position i)
                        correlated_features.append([c, col])
        
        print(f'Number of features after correlation thresholding: {len(data_X.columns) - len(to_drop)}')
        
        reduced_data_X = data_X.drop(to_drop, axis=1)
        
        # * No feature is anymore correlated with each other above the threshold value
        plt.figure(figsize=(16, 9))
        sns.heatmap(reduced_data_X.corr(), annot=True)
        plt.show()
        
        return reduced_data_X, correlated_features
    
    # Catboost does not support RFE, so we use a different method
    # def recursive_feature_elimination(self, data_X, data_y, database, best_param):
        
    #     alpha = best_param['alpha_rfe']
        
    #     # Use MultiOutputRegressor to select the features for all target variables at once (0.1 original alpha value)
    #     # selector = SelectFromModel(Lasso(alpha=alpha, max_iter=10000))
    #     model = Lasso(alpha=alpha, max_iter=10000)
    #     # model = xgb.XGBRegressor()
        
    #     # Create the RFE object and compute a cross-validated score.
    #     # cv = LeaveOneOut()
    #     # , min_features_to_select=5
    #     rfecv = RFECV(estimator=model, step=5, cv=LeaveOneOut(), n_jobs=-1, verbose=0, scoring='neg_mean_squared_error')
    #     rfecv.fit(data_X, data_y)
        
    #     print("Optimal number of features : %d" % rfecv.n_features_)
        
    #     # Plot number of features VS. cross-validation scores
    #     plt.figure()
    #     plt.xlabel("Number of features selected")
    #     plt.ylabel("Cross validation score (nb of correct classifications)")
    #     plt.plot(range(1, len(rfecv.cv_results_['mean_test_score']) + 1), rfecv.cv_results_['mean_test_score'])
    #     plt.tight_layout()
    #     plt.show()
        
    #     # Get the selected features
    #     selected_features = data_X.columns[rfecv.get_support()]
        
    #     # Save the selected features as pickle file
    #     # with open(os.path.join(PERMA_MODEL_RESULTS_DIR, self.database_name + "_selected_features.pkl"), "wb") as f:
    #     #     pkl.dump(selected_features, f)
        
    #     # Create a new DataFrame with only the kept features
    #     reduced_data_X = data_X[selected_features]
        
    #     return reduced_data_X
    
    # Loop over all target variables and select the features for each one
    def recursive_feature_elimination(self, data_X, data_y, database, best_param):
        
        alpha = best_param['alpha_rfe']
        
        perma_feature_list = []
        
        reduced_data_X_features = set()
        
        for i, col in enumerate(data_y.columns):
            model = Lasso(alpha=alpha, max_iter=10000)
            
            data_y_i = data_y.iloc[:, i]
            
            scoring = 'neg_mean_absolute_error'
            # scoring = 'neg_mean_squared_error'
            # scoring = 'neg_root_mean_squared_error'
            # scoring = 'r2'
            # Calculate cv factor, so every eval set has the length of 2 samples
            # cv_factor = int(len(data_y_i) / 2)
            # cv = KFold(n_splits=cv_factor, shuffle=True, random_state=42)
            cv = LeaveOneOut()
            rfecv = RFECV(estimator=model, step=1, cv=cv, n_jobs=-1, verbose=0, scoring=scoring, min_features_to_select=1)
            rfecv.fit(data_X, data_y_i)
            
            print(f'Optimal number of features for {col}: {rfecv.n_features_}')
            
            # Get the selected features
            selected_features = data_X.columns[rfecv.get_support()]
            
            # Create a new DataFrame with only the kept features
            # reduced_data_X_i = data_X[selected_features]

            # reduced_data_X = pd.concat([reduced_data_X, reduced_data_X_i], axis=1)
            
            # Add feature names to set (to remove duplicates)
            reduced_data_X_features.update(selected_features)
            # selected_features = sorted(selected_features)
            
            # Create a dataframe (select the features from data_X) and append it to the list
            perma_feature_list.append(selected_features)
            
            # TODO: could also store them just once
            filename = database + "_selected_features_" + str(col) + ".pkl"
            with open(os.path.join(PERMA_MODEL_RESULTS_DIR, database + "_" + str(col) + "_selected_features.pkl"), "wb") as f:
                pkl.dump(reduced_data_X_features, f)
            
            # Plot number of features VS. cross-validation scores
            # plt.figure()
            # plt.xlabel("Number of features selected")
            # plt.ylabel("Cross validation score")
            # plt.plot(range(1, len(rfecv.cv_results_['mean_test_score']) + 1), rfecv.cv_results_['mean_test_score'])
            # plt.tight_layout()
            # plt.show()

        print(f'Number of features after recursive feature elimination: {len(reduced_data_X_features)}')

        # Sort the features by alphabetical order
        reduced_data_X_features = sorted(reduced_data_X_features)

        # Create a new DataFrame with only the kept features
        reduced_data_X = data_X[reduced_data_X_features]
        
        # Save the selected features as pickle file
        # with open(os.path.join(PERMA_MODEL_RESULTS_DIR, database + "_selected_features.pkl"), "wb") as f:
        #     pkl.dump(reduced_data_X_features, f)

        return reduced_data_X, perma_feature_list
            
