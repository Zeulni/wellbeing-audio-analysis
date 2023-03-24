import os
import pandas as pd
import numpy as np

from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA


class SampleReduction():
    def __init__(self) -> None:
        pass
    
    # LOF works well for high dimensional data
    # For hints see: https://towardsdatascience.com/4-machine-learning-techniques-for-outlier-detection-in-python-21e9cfacb81d
    def remove_outliers(self, data_X, data_y) -> pd:
        
        # Extract the features
        X = data_X.values

        # Set LOF parameters for feature outlier detection
        n_neighbors = 20
        contamination = 0.03
        
        # Perform PCA for a more stable outlier detection (will just be used to find the indices, transformed data will not be used)
        # But it is not necessary to perform PCA for LOF, as no difference (at least not for my datasets)
        pca = PCA(n_components=8)
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