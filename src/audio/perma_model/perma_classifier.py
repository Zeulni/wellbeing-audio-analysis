import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import LeaveOneOut

from sklearn.metrics import balanced_accuracy_score

from src.audio.utils.constants import PERMA_MODEL_RESULTS_DIR

class PermaClassifier:
    def __init__(self, data_X_train, data_X_test, data_y_train, data_y_test, perma_feature_list, database) -> None:
        
        self.perma_feature_list = perma_feature_list
        self.database = database
        
        self.data_X_train = data_X_train
        self.data_X_test = data_X_test
        self.data_y_train = data_y_train
        self.data_y_test = data_y_test     
    
        self.number_of_classes = 4
        self.labels = [i for i in range(self.number_of_classes)]   
        # self.labels = ['q1', 'q2', 'q3', 'q4']
    
        self.bin_perma_pillars()
        
        self.knn_param_grid = {
            'n_neighbors': [3, 5, 7, 9, 11, 13, 15],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan']
        }
        self.knn_class_model = KNeighborsClassifier()
        
        self.random_forest_param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 5]
        }
        self.random_forest_class_model = RandomForestClassifier(class_weight='balanced')
        
        # TODO: also class_weight option for xgboost?
        # * XGBoost Model
        self.xgboost_param_grid = {
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 5, 7]
        }
        # Performs better with squarrederror than absolute error
        self.xgboost_class_model = xgb.XGBClassifier(objective='multi:softmax', num_class=self.number_of_classes)
        
        # * CatBoost Model
        self.catboost_param_grid = {
            'learning_rate': [0.01, 0.1],
            'depth': [3, 5, 7]
        }
        
        self.catboost_class_model = CatBoostClassifier(loss_function='MultiClass', classes_count=self.number_of_classes, verbose=False, save_snapshot=False, allow_writing_files=False, train_dir=str(PERMA_MODEL_RESULTS_DIR))
        
        # Create the lists with params and models
        self.model_name_list = ["random_forest", "xgboost", "catboost", "knn"]
        self.model_param_grid_list = [self.random_forest_param_grid, self.xgboost_param_grid, self.catboost_param_grid, self.knn_param_grid]
        self.class_model_list = [self.random_forest_class_model, self.xgboost_class_model, self.catboost_class_model, self.knn_class_model]
        
        # self.model_name_list = ["xgboost", "knn"]
        # self.model_param_grid_list = [self.xgboost_param_grid, self.knn_param_grid]
        # self.class_model_list = [self.xgboost_class_model, self.knn_class_model]
        
        self.baseline_comp_dict = {"P": {"baseline": [], "prediction": []},
                                   "E": {"baseline": [], "prediction": []},
                                   "R": {"baseline": [], "prediction": []},
                                   "M": {"baseline": [], "prediction": []},
                                   "A": {"baseline": [], "prediction": []}}
    
    # Bin the data
    def bin_perma_pillars(self):
     
        percentiles = np.linspace(0, 100, num=(self.number_of_classes+1))
        
        # Iterate through the lenght of the data_y_train
        for perma_pillar in self.data_y_train:
            perma_pillar_data = self.data_y_train[perma_pillar]
            print(perma_pillar_data.value_counts().sort_index())
            bin_edges = np.percentile(perma_pillar_data, percentiles)
            bin_edges[0] = -np.inf
            bin_edges[-1] = np.inf
            
            # Print the distribution of each class in the train set
            print("Distribution of classes in train set for " + perma_pillar + ":")
            print(pd.cut(perma_pillar_data, bins=bin_edges, labels=self.labels).value_counts().sort_index())
            
            # Replace the values in the data_y_train with the binned values
            self.data_y_train[perma_pillar] = pd.cut(perma_pillar_data, bins=bin_edges, labels=self.labels)
            # Replace the values in the data_y_test with the binned values
            self.data_y_test[perma_pillar] = pd.cut(self.data_y_test[perma_pillar], bins=bin_edges, labels=self.labels)
            
    def train_multiple_models_per_pillar(self):
        
        self.train_ind_models()
        
        self.plot_baseline_bar_plot_comparison()
    
    def train_ind_models(self):
        
        self.best_models = []
        self.best_models_names = []        
        
        # Loop over every pillar
        for perma_i in range(self.data_y_train.shape[1]): 
            pillar_models = []
            pillar_best_scores = []
            for model_i, class_model in enumerate(self.class_model_list):
            
                param_grid = self.model_param_grid_list[model_i]
                model_name = self.model_name_list[model_i]     
                # cv=LeaveOneOut()
                grid_search = GridSearchCV(class_model, param_grid, cv=LeaveOneOut(), scoring='balanced_accuracy', verbose=0, n_jobs=-1, refit='balanced_accuracy')
                
                # Convert the data to DMatrix
                # dtrain = xgb.DMatrix(self.data_X_train[self.perma_feature_list[perma_i]], label=self.data_y_train.iloc[:, perma_i])
                
                grid_search.fit(self.data_X_train[self.perma_feature_list[perma_i]], self.data_y_train.iloc[:, perma_i])   
                
                pillar_models.append(grid_search.best_estimator_)
                pillar_best_scores.append(grid_search.best_score_)
                
                print("Best score for " + model_name + " : ", round(grid_search.best_score_,3))
                
            best_model_i = np.argmax(pillar_best_scores)
            self.best_models.append(pillar_models[best_model_i])
            self.best_models_names.append(self.model_name_list[best_model_i])
            
            self.calc_baseline_comparison(self.best_models[perma_i], self.best_models_names[perma_i], perma_i)
        
        # TODO: for xgboost use DMatrix
            
        #     perma_pillar = str(self.data_y_train.columns[perma_i])
        #     model_file_name = self.database_name + '_' + self.best_models_names[perma_i] + '_' + perma_pillar + '_perma_model.pkl'
        #     with open(PERMA_MODEL_RESULTS_DIR / model_file_name, 'wb') as f:
        #         pkl.dump(self.best_models[perma_i], f)
        
        
    def calc_baseline_comparison(self, class_model, model_name, y_i):
        
        perma_pillar = str(self.data_y_train.columns[y_i])
        data_y_train = self.data_y_train.iloc[:, y_i]
        data_y_test = self.data_y_test.iloc[:, y_i]
        
        # Baseline: just always predict the majority class
        majority_class = data_y_train.value_counts().idxmax()
        y_pred_baseline = np.tile(majority_class, (len(data_y_test), 1))
        
        # Calculate balanced_accuracy as metric (same as above)
        baseline_bal_acc = balanced_accuracy_score(data_y_test, y_pred_baseline)
        
        print (model_name + " Test Set Baseline Balanced Acc:", round(baseline_bal_acc,3))
        self.baseline_comp_dict[perma_pillar]["baseline"].append(baseline_bal_acc)
            
        # Calculate the balanced_accuracy of the model
        y_pred = class_model.predict(self.data_X_test[self.perma_feature_list[y_i]])
        
        model_bal_acc = balanced_accuracy_score(data_y_test, y_pred)

        print (model_name + " Test Set Prediction Balanced Acc:", round(model_bal_acc,3))
        self.baseline_comp_dict[perma_pillar]["prediction"].append(model_bal_acc)
        
    def plot_baseline_bar_plot_comparison(self):
        
        # Initialize empty lists to store the model names and average values
        perma_pillars = []
        baseline_avgs = []
        prediction_avgs = []

        # Iterate over the keys in the baseline_comp_dict dictionary to extract the model names and values
        for perma_pillar, values in self.baseline_comp_dict.items():
            perma_pillars.append(perma_pillar)
            baseline_avgs.append(np.mean(values["baseline"]))
            prediction_avgs.append(np.mean(values["prediction"]))

        # Set up the plot
        plt.rcParams['font.size'] = 16
        fig, ax = plt.subplots()
        bar_width = 0.35
        opacity = 0.8
        index = np.arange(len(perma_pillars))

        # Create the bars for the baseline values
        rects1 = ax.bar(index, baseline_avgs, bar_width,
                        alpha=opacity,
                        color='b',
                        label='Baseline')

        # Create the bars for the prediction values
        rects2 = ax.bar(index + bar_width, prediction_avgs, bar_width,
                        alpha=opacity,
                        color='g',
                        label='Prediction')
        
        # Show over each bars the name of the best model (in self.best_models_names), the names are tilted 90 degrees
        for i, rect in enumerate(rects2):
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                    self.best_models_names[i],
                    ha='center', va='bottom', rotation=90, fontsize=10)
       
        # Add labels, title and legend to the plot
        ax.set_xlabel('PERMA Pillars')
        ax.set_ylabel('Baseline vs Prediction Balanced Acc for best models')
        ax.set_title('Baseline and Prediction Averages by PERMA Pillar')
        ax.set_xticks(index + bar_width / 2)
        ax.set_xticklabels(perma_pillars, fontsize=20)
        ax.legend(loc="lower right")

        # Show the plot
        plt.show()