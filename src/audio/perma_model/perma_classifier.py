import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import shap

import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import LeaveOneOut
from sklearn.base import clone

from sklearn.metrics import balanced_accuracy_score

from src.audio.utils.constants import PERMA_MODEL_RESULTS_DIR

class PermaClassifier:
    def __init__(self, data_X_train, data_X_test, data_y_train, data_y_test, perma_feature_list, database, number_classes) -> None:
        
        self.perma_feature_list = perma_feature_list
        self.database = database
        
        self.data_X_train = data_X_train
        self.data_X_test = data_X_test
        self.data_y_train = data_y_train
        self.data_y_test = data_y_test     
    
        self.number_of_classes = number_classes
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
        
        # * XGBoost Model
        self.xgboost_param_grid = {
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 5, 7]
        }
        self.xgboost_class_model = xgb.XGBClassifier(objective='multi:softmax', num_class=self.number_of_classes)
        
        # * CatBoost Model
        self.catboost_param_grid = {
            'learning_rate': [0.01, 0.1],
            'depth': [3, 5, 7]
        }
        
        self.catboost_class_model = CatBoostClassifier(loss_function='MultiClass', classes_count=self.number_of_classes, verbose=False, save_snapshot=False, allow_writing_files=False, train_dir=str(PERMA_MODEL_RESULTS_DIR))
        
        # Create the lists with params and models
        self.model_name_list = ["Random Forest", "XGBoost", "CatBoost", "k-NN"]
        self.model_param_grid_list = [self.random_forest_param_grid, self.xgboost_param_grid, self.catboost_param_grid, self.knn_param_grid]
        self.class_model_list = [self.random_forest_class_model, self.xgboost_class_model, self.catboost_class_model, self.knn_class_model]
        
        # self.model_name_list = ["knn"]
        # self.model_param_grid_list = [self.knn_param_grid]
        # self.class_model_list = [self.knn_class_model]
        
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
            # print(perma_pillar_data.value_counts().sort_index())
            bin_edges = np.percentile(perma_pillar_data, percentiles)
            bin_edges[0] = -np.inf
            bin_edges[-1] = np.inf
            
            # Print the distribution of each class in the train set
            # print("Distribution of classes in train set for " + perma_pillar + ":")
            # print(pd.cut(perma_pillar_data, bins=bin_edges, labels=self.labels).value_counts().sort_index())
            
            # Replace the values in the data_y_train with the binned values
            self.data_y_train[perma_pillar] = pd.cut(perma_pillar_data, bins=bin_edges, labels=self.labels)
            # Replace the values in the data_y_test with the binned values
            self.data_y_test[perma_pillar] = pd.cut(self.data_y_test[perma_pillar], bins=bin_edges, labels=self.labels)
            
    def train_multiple_models_per_pillar(self):
        
        self.train_ind_models()
        self.plot_and_save_shap_values()
        self.plot_baseline_bar_plot_comparison()
        # print("Best params: ", self.best_params)
        # print("---------------------")
    
    def train_ind_models(self):
        
        if self.load_results():
            return
        
        self.best_models = []
        self.best_models_names = []    
        self.best_params = []    
        
        # Loop over every pillar
        for perma_i in range(self.data_y_train.shape[1]): 
            pillar_models = []
            pillar_best_scores = []
            pillar_best_params = []
            for model_i, class_model_template in enumerate(self.class_model_list):
                
                class_model = clone(class_model_template)
            
                param_grid = self.model_param_grid_list[model_i]
                model_name = self.model_name_list[model_i]     
                # cv=LeaveOneOut()
                grid_search = GridSearchCV(class_model, param_grid, cv=LeaveOneOut(), scoring='balanced_accuracy', verbose=0, n_jobs=-1, refit='balanced_accuracy')
                
                # Convert the data to DMatrix
                # dtrain = xgb.DMatrix(self.data_X_train[self.perma_feature_list[perma_i]], label=self.data_y_train.iloc[:, perma_i])
                
                grid_search.fit(self.data_X_train[self.perma_feature_list[perma_i]], self.data_y_train.iloc[:, perma_i])   
                
                pillar_models.append(grid_search.best_estimator_)
                pillar_best_scores.append(grid_search.best_score_)
                pillar_best_params.append(grid_search.best_params_)
                
                print("Best score for " + model_name + " : ", round(grid_search.best_score_,3))
                
            best_model_i = np.argmax(pillar_best_scores)
            self.best_models.append(pillar_models[best_model_i])
            self.best_models_names.append(self.model_name_list[best_model_i])
            self.best_params.append(pillar_best_params[best_model_i])
            
            self.calc_baseline_comparison(self.best_models[perma_i], self.best_models_names[perma_i], perma_i)
        
        self.save_results()
        
        
    def load_results(self):
            
        best_models_filename = os.path.join(PERMA_MODEL_RESULTS_DIR, "classification_n" + str(self.number_of_classes) + "_" + self.database + "_best_models.pkl")
        best_models_names_filename = os.path.join(PERMA_MODEL_RESULTS_DIR, "classification_n" + str(self.number_of_classes) + "_" + self.database + "_best_model_names.pkl")
        best_params_filename = os.path.join(PERMA_MODEL_RESULTS_DIR, "classification_n" + str(self.number_of_classes) + "_" + self.database + "_best_params.pkl")
        baseline_comp_dict_filename = os.path.join(PERMA_MODEL_RESULTS_DIR, "classification_n" + str(self.number_of_classes) + "_" + self.database + "_baseline_comp_dict.pkl")
        if os.path.exists(best_models_filename) and os.path.exists(best_models_names_filename) and os.path.exists(best_params_filename) and os.path.exists(baseline_comp_dict_filename):
            # Load the best models as pickle file
            with open(best_models_filename, "rb") as f:
                self.best_models = pkl.load(f)
                
            # Load the best model names as pickle file
            with open(best_models_names_filename, "rb") as f:
                self.best_models_names = pkl.load(f)
                
            # Load the best params as pickle file
            with open(best_params_filename, "rb") as f:
                self.best_params = pkl.load(f)
                
            # Load the baseline comparison dict as pickle file
            with open(baseline_comp_dict_filename, "rb") as f:
                self.baseline_comp_dict = pkl.load(f)
            
            return True
                
        else:
            return False
        
    def save_results(self):
        
        # Save the best models as pickle file
        with open(os.path.join(PERMA_MODEL_RESULTS_DIR, "classification_n" + str(self.number_of_classes) + "_" + self.database + "_best_models.pkl"), "wb") as f:
            pkl.dump(self.best_models, f)
            
        # Save the best model names as pickle file
        with open(os.path.join(PERMA_MODEL_RESULTS_DIR, "classification_n" + str(self.number_of_classes) + "_"  + self.database + "_best_model_names.pkl"), "wb") as f:
            pkl.dump(self.best_models_names, f)
            
        # Save the best params as pickle file
        with open(os.path.join(PERMA_MODEL_RESULTS_DIR, "classification_n" + str(self.number_of_classes) + "_"  + self.database + "_best_params.pkl"), "wb") as f:
            pkl.dump(self.best_params, f)
            
        # Save the self.baseline_comp_dict
        with open(os.path.join(PERMA_MODEL_RESULTS_DIR, "classification_n" + str(self.number_of_classes) + "_"  + self.database + "_baseline_comp_dict.pkl"), "wb") as f:
            pkl.dump(self.baseline_comp_dict, f)
        
        
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
        plt.rcParams['font.size'] = 12
        fig, ax = plt.subplots()
        bar_width = 0.35
        opacity = 0.8
        index = np.arange(len(perma_pillars))

        # Create the bars for the baseline values
        rects1 = ax.bar(index, baseline_avgs, bar_width,
                        alpha=opacity,
                        color='#333333',
                        label='Baseline')

        # Create the bars for the prediction values
        rects2 = ax.bar(index + bar_width, prediction_avgs, bar_width,
                        alpha=opacity,
                        color='#999999',
                        label='Prediction')
        
        # Show over each bars the name of the best model (in self.best_models_names), the names are tilted 90 degrees
        for i, rect in enumerate(rects2):
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2., 0.4*height,
                    self.best_models_names[i],
                    ha='center', va='bottom', rotation=90, fontsize=12)

       
        # Add labels, title and legend to the plot
        ax.set_xlabel('PERMA pillars', fontsize=14)
        ax.set_ylabel('Balanced accuracy', fontsize=14)
        # ax.set_title('Baseline vs. Prediction Balanced Accuracy by PERMA Pillar (on Test Set)', fontsize=12)
        ax.set_xticks(index + bar_width / 2)
        ax.set_xticklabels(perma_pillars, fontsize=14)
        ax.legend(loc="lower right")
        
        plt.tight_layout()
        plt.grid(True, alpha=0.5)
        
        # Adjust plot to the left and right
        plt.subplots_adjust(left=0.15, right=0.90, top=0.9, wspace=1.8, hspace=0.3)
        
        filename = os.path.join(PERMA_MODEL_RESULTS_DIR, "classification_test_set_eval_n"  + str(self.number_of_classes) + "_" + self.database + ".png")
        plt.savefig(filename, dpi=600)
        
        # Show the plot
        plt.show()
    
    def plot_rel_baseline_comp_n_classes(self):
        
        baseline_comp_dict_n2_filename = os.path.join(PERMA_MODEL_RESULTS_DIR, "classification_n2_" + self.database + "_baseline_comp_dict.pkl")
        baseline_comp_dict_n3_filename = os.path.join(PERMA_MODEL_RESULTS_DIR, "classification_n3_" + self.database + "_baseline_comp_dict.pkl")
        baseline_comp_dict_n4_filename = os.path.join(PERMA_MODEL_RESULTS_DIR, "classification_n4_" + self.database + "_baseline_comp_dict.pkl")
        
        with open(baseline_comp_dict_n2_filename, "rb") as f:
            baseline_comp_dict_n2 = pkl.load(f)
        with open(baseline_comp_dict_n3_filename, "rb") as f:
            baseline_comp_dict_n3 = pkl.load(f)
        with open(baseline_comp_dict_n4_filename, "rb") as f:
            baseline_comp_dict_n4 = pkl.load(f)
        
        # Calculate for each dictionary the ratio of the prediction to the baseline and store it in a new dictionary
        baseline_comp_dict_n2_ratio = {}
        baseline_comp_dict_n3_ratio = {}
        baseline_comp_dict_n4_ratio = {}

        for perma_pillar, values in baseline_comp_dict_n2.items():
            baseline_comp_dict_n2_ratio[perma_pillar] = np.mean(values["prediction"]) / np.mean(values["baseline"])
        for perma_pillar, values in baseline_comp_dict_n3.items():
            baseline_comp_dict_n3_ratio[perma_pillar] = np.mean(values["prediction"]) / np.mean(values["baseline"]) 
        for perma_pillar, values in baseline_comp_dict_n4.items(): 
            baseline_comp_dict_n4_ratio[perma_pillar] = np.mean(values["prediction"]) / np.mean(values["baseline"])
            
        # Create a bar plot to plot the ratio of the prediction to the baseline for n_classes = 2, 3 and 4 and PERMA pillars (x-axis)
        plt.rcParams['font.size'] = 12
        fig, ax = plt.subplots()
        bar_width = 0.20
        opacity = 0.8
        index = np.arange(len(baseline_comp_dict_n2_ratio.keys()))

        # Create the bars for the n_classes = 2 values
        rects1 = ax.bar(index, baseline_comp_dict_n2_ratio.values(), bar_width,
                        alpha=opacity,
                        color='#000000',
                        label='number of classes = 2')
        
        # Create the bars for the n_classes = 3 values
        rects2 = ax.bar(index + bar_width, baseline_comp_dict_n3_ratio.values(), bar_width,
                        alpha=opacity,
                        color='#666666',
                        label='number of classes = 3')
        
        # Create the bars for the n_classes = 4 values
        rects3 = ax.bar(index + 2*bar_width, baseline_comp_dict_n4_ratio.values(), bar_width,
                        alpha=opacity,
                        color='#bbbbbb',
                        label='number of classes = 4')
    
        # Add labels, title and legend to the plot
        ax.set_xlabel('PERMA pillars', fontsize=14)
        ax.set_ylabel('Prediction/baseline ratio', fontsize=14)
        # ax.set_title('Prediction/Baseline Ratio per #classes by PERMA Pillar (on Test Set)', fontsize=12)
        ax.set_xticks(index + bar_width / 2)
        ax.set_xticklabels(baseline_comp_dict_n2_ratio.keys(), fontsize=14)
        ax.legend(loc="lower right")
        
        plt.tight_layout()
        plt.grid(True, alpha=0.5)
        # plt.subplots_adjust(left=0.10, right=0.96, top=0.9, wspace=1.8, hspace=0.3)
        
        # Calculate the average ratio for each number of classes
        # Extract values from a dictionary and store them in a numpy array
        n2_avg = np.mean(list(baseline_comp_dict_n2_ratio.values()))
        n3_avg = np.mean(list(baseline_comp_dict_n3_ratio.values()))
        n4_avg = np.mean(list(baseline_comp_dict_n4_ratio.values()))
        
        
        # Save the plot in PERMA_MODEL_RESULTS_DIR
        filename = os.path.join(PERMA_MODEL_RESULTS_DIR, "classification_test_set_nclass_comp_" + self.database + ".png")
        plt.savefig(filename, dpi=600)

        # Show the plot
        plt.show()
        
    def sim_func(self, x1, x2):
        return 1 / (np.linalg.norm(x1 - x2) + 1e-6)
        
    def plot_and_save_shap_values(self):        
    
        # Only plot shap values for the top 5 important features for each estimator
        for i, reg_model in enumerate(self.best_models):
            model_name = self.best_models_names[i]
            if model_name == 'k-NN':
                explainer = shap.KernelExplainer(reg_model.predict_proba, self.data_X_train[self.perma_feature_list[i]], link="logit", sim_func=self.sim_func)
                feature_names = reg_model.feature_names_in_
            if model_name == 'CatBoost' or model_name == 'XGBoost' or model_name == 'RandomForest':
                explainer = shap.TreeExplainer(reg_model)
                feature_names = reg_model.feature_names_in_
            # elif model_name == 'Lasso' or model_name == 'Ridge':
            #     explainer = shap.Explainer(reg_model, self.data_X_train[self.perma_feature_list[i]])
            #     feature_names = reg_model.feature_names_in_
                
            shap_values = explainer.shap_values(self.data_X_train[self.perma_feature_list[i]])
            # feature_names = [" "]
            
            # Create a list of empty feature names depending on the number of features (in perma_feature_list[i])
            feature_names = [" " for i in range(len(self.perma_feature_list[i]))]
            
            # Replace all the nan values with 0
            # Replace inf values with a large number
            shap_values = np.where(np.isneginf(shap_values), -1e10, shap_values)
            shap_values = np.where(np.isposinf(shap_values), 1e10, shap_values)
            shap_values = list(np.nan_to_num(shap_values))

            # TODO: has to be changed for the classification case
            # shap_class = 2
            # shap_values = shap_values[shap_class]
            shap.summary_plot(shap_values, self.data_X_train[self.perma_feature_list[i]], feature_names = feature_names, show=False)
            plt.xlabel("mean(|SHAP value|)")
            # plt.xlabel("SHAP Values")
            perma_pillar = str(self.data_y_train.columns[i])
            # plt.title(perma_pillar, fontsize=20, fontweight='bold')
            # plt.savefig(PERMA_MODEL_RESULTS_DIR / f'{self.database}_classification_shap_values_{self.number_of_classes}_class_{shap_class}_{perma_pillar}.png', dpi=600)
            plt.savefig(PERMA_MODEL_RESULTS_DIR / f'{self.database}_classification_shap_values_{self.number_of_classes}_class_{perma_pillar}.png', dpi=600)
            # plt.show()
            plt.clf()