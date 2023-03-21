from numpy import mean, std, absolute
import pickle as pkl
import matplotlib.pyplot as plt
import shap

from catboost import CatBoostRegressor, cv, Pool
import xgboost as xgb

from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold

from src.audio.utils.constants import PERMA_MODEL_DIR

class PermaRegressor:
    def __init__(self, data_X, data_y, database_name) -> None:
        self.data_X = data_X
        self.data_y = data_y
        self.database_name = database_name
        
    def train_model(self, regr, model_name):
        
        # TODO: Also test LOOCV?
        cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=1)
        n_scores = cross_val_score(regr, self.data_X, self.data_y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
        # force the scores to be positive
        n_scores = absolute(n_scores)
        # summarize performance
        print('MAE: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
        
        # TODO: If using "fit" after cross validation, will the results from cv be overwritten?
        regr.fit(self.data_X, self.data_y)
        
        # Make a test prediction with dummy data
        # test_data = self.data_X.iloc[0:1]
        # print(regr.predict(test_data))
        
        self.plot_and_save_feature_importance(regr, model_name)
        self.plot_and_save_shap_values(regr, model_name)
    
        # Save models dict to pickle file
        model_file_name = self.database_name + '_' + model_name + '_perma_model.pkl'
        with open(PERMA_MODEL_DIR / model_file_name, 'wb') as f:
            pkl.dump(regr, f)       
    
    def catboost_train(self):
        
        # TODO: Adapt regularization through depth
        params = {'loss_function':'RMSE', 'depth': 5, 'verbose': False, "save_snapshot": False, "allow_writing_files": False, "train_dir": str(PERMA_MODEL_DIR)}
        regr = MultiOutputRegressor(CatBoostRegressor(**params))
        self.train_model(regr, 'catboost')


    def xgboost_train(self):
        # Define XGBoost hyperparameters
        # TODO: Adapt regularization through depth
        params = {'objective': 'reg:squarederror', 'max_depth': 5, 'learning_rate': 0.1}    
        regr = MultiOutputRegressor(xgb.XGBRegressor(**params))
        self.train_model(regr, 'xgboost')
        
    def plot_and_save_feature_importance(self, regr, model_name):
        
        # Define the size of the plot
        plt.figure(figsize=(16, 8))
        
        # Loop over every estimator and plot the feature importance as a subplot
        for i, estimator in enumerate(regr.estimators_):
            plt.subplot(2, 3, i+1)
            sorted_feature_importance = estimator.feature_importances_.argsort()
            plt.barh(self.data_X.columns[sorted_feature_importance], 
                    estimator.feature_importances_[sorted_feature_importance], 
                    color='turquoise')
            plt.xlabel(model_name + " Feature Importance")
            plt.title(self.data_y.columns[i])
            plt.tight_layout()
        
        plt.savefig(PERMA_MODEL_DIR / f'{self.database_name}_{model_name}_feature_importance.png')
        plt.show()
        plt.clf()
        
        
    # TODO: How to interpret the shap values?
    def plot_and_save_shap_values(self, regr, model_name):        
        
        # Only plot shap values for the top 5 important features for each estimator
        for i, estimator in enumerate(regr.estimators_):
            explainer = shap.TreeExplainer(estimator)
            shap_values = explainer.shap_values(self.data_X)
            
            show_last_n_features = 5
            sorted_feature_importance = estimator.feature_importances_.argsort()[-show_last_n_features:]
            columns_shap_values = self.data_X.columns[sorted_feature_importance]
            shap_values = shap_values[:, sorted_feature_importance]

            shap.summary_plot(shap_values, self.data_X[columns_shap_values], feature_names = columns_shap_values, show=False)
            # plt.xlabel("SHAP Values")
            plt.title(self.data_y.columns[i])
            plt.savefig(PERMA_MODEL_DIR / f'{self.database_name}_{model_name}_shap_values_{self.data_y.columns[i]}.png')
            plt.show()
            plt.clf()