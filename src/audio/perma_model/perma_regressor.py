import pickle as pkl
import matplotlib.pyplot as plt
import shap

from catboost import CatBoostRegressor, cv, Pool

from src.audio.utils.constants import PERMA_MODEL_DIR

class PermaRegressor:
    def __init__(self, data_y_X_dict) -> None:
        self.data_y_X_dict = data_y_X_dict
    
    def catboost_train(self):
        
        models = {}
        # Use RMSE and MAE
        # TODO: Regulization through depth
        # params = {'loss_function':'RMSE', 'eval_metric':'MAE', 'verbose': 200, 'random_seed': 13, 'depth': 5}
          
        # params = {'loss_function':'RMSE', 'depth': 5, 'verbose': False, "save_snapshot": False}
        # Prevent folder "catboost_info" from being created
        params = {'loss_function':'RMSE', 'depth': 5, 'verbose': False, "save_snapshot": False, "allow_writing_files": False, "train_dir": str(PERMA_MODEL_DIR)}
        
        for target, data in self.data_y_X_dict.items():
        
            y = data[0]
            X = data[1]

            X_pool = Pool(data=X, label=y)
            
            # scores = cv(pool=X_pool, params=params, fold_count=4, seed=13,
            #         shuffle=True)

            scores = cv(pool=X_pool, params=params, fold_count=4, seed=13,
                    shuffle=True)
            
            print('RMSE: {}'.format(scores['test-RMSE-mean'].min()))
            
            # Save model
            model = CatBoostRegressor(**params)
            model.fit(X_pool)
            
            self.plot_feature_importance(model, X)
            self.plot_shape_values(model, X)
            
            models[target] = model
    
        # Save models dict to pickle file
        with open(PERMA_MODEL_DIR / 'perma_catboost_models.pkl', 'wb') as f:
            pkl.dump(models, f)
        
    def plot_feature_importance(self, model, X):
        sorted_feature_importance = model.feature_importances_.argsort()
        plt.barh(X.columns[sorted_feature_importance], 
                model.feature_importances_[sorted_feature_importance], 
                color='turquoise')
        plt.xlabel("CatBoost Feature Importance")
        plt.show()
        
    # TODO: How to interpret the shap values?
    def plot_shape_values(self, model, X):
        sorted_feature_importance = model.feature_importances_.argsort()
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        shap.summary_plot(shap_values, X, feature_names = X.columns[sorted_feature_importance])
    
    # examples e.g. for Cross Validation or how to save model:
    # https://catboost.ai/en/docs/concepts/python-usages-examples
    