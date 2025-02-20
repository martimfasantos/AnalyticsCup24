import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score
from imblearn.over_sampling import SMOTE, RandomOverSampler

# Parse terminal input
FLAG = ''
valid_flags = ('missing_values', 'outliers', 'scaling', 'feature_selection')
if len(sys.argv) == 2 and sys.argv[1] in valid_flags:
    FLAG = sys.argv[1]
else:
    print("Invalid format, try:  python eval_aux.py [missing_values|outliers|scaling|feature_selection]")
    exit(1)
    
# Folder path
dir_path = f'datasets/{FLAG}/'

# List to store files
file_names = []
file_paths = []

# Iterate directory
for file in os.listdir(dir_path):
    # check if current path is a file
    if os.path.isfile(os.path.join(dir_path, file)):
        file_name = os.path.splitext(file)[0]
        file_names.append(file_name)
        file_paths.append(f'datasets/{FLAG}/{file_name}')

print("Evaluating datasets: ", file_names)
print("-" * 20)

# ---------------------------------------------------------------------------

# target variable
target = 'Like'

NUMBER_ITERATIONS = 10

for i in range(len(file_names)):
    mean_balanced_accuracy = 0
    for j in range(NUMBER_ITERATIONS):
        file_path = file_paths[i]
        
        # read data
        df = pd.read_csv(f'{file_path}.csv', low_memory=False)
        # remove index column
        index_column = df.columns[0]
        df = df.drop([index_column], axis=1)

        # Drop TestSetId column and Like NaN rows (no way to know if they liked or not)
        df.drop('TestSetId', axis=1, inplace=True)
        df.dropna(subset=[target], inplace=True)
        
        # Take a random sample of 10k rows
        df_test = df.sample(n=int(df.shape[0]/10)).copy(deep=True)
        df_test.to_csv(f'df_test_aux.csv', index=False)
        df.drop(df_test.index, axis=0, inplace=True)

        # ----------------------------- #
        #           BALANCING           #
        # ----------------------------- #
        like_zero = df[df[target] == 0.0]
        like_one = df[df[target] == 1.0]

        df_one_sample = like_one.sample(len(like_zero), replace=True)
        df_zero_sample = like_zero.sample(len(like_zero))

        df = pd.concat([df_zero_sample, df_one_sample], axis=0)
        # ----------------------------- #

        y = df.pop(target).values
        X = df.values

        X_train, X_test, y_train, y_test = \
            train_test_split(X, y,
                        test_size=0.3, 
                        shuffle=True,
                        random_state=3)
        
        # Here, we want to find the best classifier. As candidates, we consider
        #   1. LogisticRegression
        #   2. RandomForestClassifier
        #   3. GradientBoostingClassifier
        #   4. HistGradientBoostingClassifier
        #   5. AdaBoostClassifier
        #   6. MLPClassifier
            
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier, AdaBoostClassifier
        from sklearn.neural_network import MLPClassifier
        from sklearn.decomposition import PCA

        from sklearn.pipeline import Pipeline
        from sklearn.model_selection import GridSearchCV


        model_logistic_regression = LogisticRegression(max_iter=300)
        model_random_forest = RandomForestClassifier()
        model_gradient_boosting = GradientBoostingClassifier()
        model_adaboost = AdaBoostClassifier()
        model_hist_gradient_boosting = HistGradientBoostingClassifier()
        model_neural_network = MLPClassifier(max_iter=200)


        pipeline = Pipeline(steps=[("model", None)])

        parameter_grid_preprocessing = {
        # Empty on purpose (no preprocessing)
        }

        # NOTE: Logistic Regression does not perform as well as the other models
        parameter_grid_logistic_regression = {
        "model" : [model_logistic_regression],
        "model__C" : [0.1, 1, 10],  # inverse regularization strength
        }

        parameter_grid_gradient_boosting = {
        "model" : [model_gradient_boosting],
        "model__n_estimators" : [150],
        "model__max_depth" : list(range(df.shape[1]-12, df.shape[1]+1)), # 4
        # "model__learning_rate" : [0.2],
        }

        # This estimator is much faster than GradientBoostingClassifier for big datasets (n_samples >= 10 000).
        parameter_grid_hist_gradient_boosting = {
        "model" : [model_hist_gradient_boosting],
        # "model__learning_rate" : [0.2], 
        "model__max_depth" : list(range(df.shape[1]-15, df.shape[1]+1)), # 4
        }

        parameter_grid_adaboost = {
        "model" : [model_adaboost],
        "model__n_estimators" : [150]
        }

        parameter_grid_random_forest = {
        "model" : [model_random_forest],
        "model__n_estimators" : [10, 20, 50],  # number of max trees in the forest
        "model__max_depth" : [5, 10, 15],
        }

        # NOTE: NN does not perform well on this dataset + takes a long time to train
        parameter_grid_neural_network = {
            "model": [model_neural_network],
            "model__hidden_layer_sizes": [(30, 30), (40, 30)],  # Example hidden layer configurations
            "model__alpha": [0.0001],  # Regularization parameter
        }


        meta_parameter_grid = [
                            # parameter_grid_logistic_regression,
                            #  parameter_grid_random_forest] #,
                            #  parameter_grid_gradient_boosting] #,
                            parameter_grid_hist_gradient_boosting] #,
                            #  parameter_grid_adaboost ] #,
                            #  parameter_grid_neural_network]

        meta_parameter_grid = [{**parameter_grid_preprocessing, **model_grid}
                            for model_grid in meta_parameter_grid]

        search = GridSearchCV(pipeline,
                            meta_parameter_grid, 
                            scoring="balanced_accuracy",
                            n_jobs=-1, 
                            cv=5,  # number of folds for cross-validation 
                            error_score="raise"
        )

        # here, the actual training and grid search happens
        search.fit(X_train, y_train.ravel())

        # print("best parameter:", search.best_params_ ,"(CV score=%0.3f)" % search.best_score_)
        # print("-" * 20)
        
        # evaluate performance of model on test set
        # print("Score on test set:", search.score(X_test, y_test.ravel()))    

        # contingency table
        ct = pd.crosstab(search.best_estimator_.predict(X_test), y_test.ravel(),
                        rownames=["pred"], colnames=["true"])
        # print(ct)
        # print("-" * 20)
        
        # ----------------------------------------------------------------- #
        # TEST with a random sample of 10k rows from the original dataset   #
        # ----------------------------------------------------------------- #

        from sklearn.metrics import balanced_accuracy_score

        # read data
        df_test = pd.read_csv('df_test_aux.csv', low_memory=False)

        def micro_service_classify_review_test(datapoint):
            # make sure the provided datapoints adhere to the correct format for model input
            
            # fetch your trained model
            model = search.best_estimator_

            # make prediction with the model
            prediction = model.predict(datapoint)

            return prediction

        # Save the Like values in a vector
        like_values = df_test['Like'].values

        # Optionally, you can drop the 'Like' column from the sampled_df if you don't need it
        df_test.drop('Like', axis=1, inplace=True)

        # make the missing predictions for the Like column
        df_test['Like'] = micro_service_classify_review_test(df_test.values)

        # Calculate balanced accuracy
        balanced_acc = balanced_accuracy_score(like_values, df_test['Like'])

        # print(f"Balanced Accuracy: {balanced_acc}")
        mean_balanced_accuracy += balanced_acc
        print(f"\t{file_names[i]} - Iteration {j+1} - Balanced Accuracy: {balanced_acc}")
    print(f"Mean Balanced Accuracy {file_names[i]}: {mean_balanced_accuracy/NUMBER_ITERATIONS}")
    print("-" * 20)