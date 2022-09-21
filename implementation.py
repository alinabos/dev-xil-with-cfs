from genericpath import isfile
from xmlrpc.client import Boolean
import dice_ml
import inquirer
import logging as log
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.neural_network import MLPClassifier
from src.TabularDatasetProcessor import TabularDatasetProcessor
from src.helper_models import init_helper_models

SEED = 42


def train_model_with_cfs(data_path, datafiles_with_header, model, epochs, threshold, automate=False, output_path=Path()):

    # validate user input
    if datafiles_with_header is False:
        log.warning("Providing a header in your data files is vital in order for the human expert to validate the counterfactual examples in an appropriate time. Try to provide a header in your files.")

    log.info(f"Number of rounds: {epochs}")

    DATA_PATH = Path() / data_path
    log.debug(f"Looking in {DATA_PATH} for data files (assuming \".data\" and \".test\" as file endings)")

    OUTPUT_PATH = Path() / output_path
    model_logfile = OUTPUT_PATH / "model_performance"
    if model_logfile.is_file():
        open(file=model_logfile, mode="w").close()

    # read dataset from data_path (assuming ".data" and ".test" as file endings)
    data_files = sorted([str(file.name) for file in Path(DATA_PATH).glob("*.data")], key=str.lower)
    data_file = data_files[0]
    log.debug(f"Found the following \".data\" files: {data_files}")

    test_files = sorted([str(file.name) for file in Path(DATA_PATH).glob("*.test")], key=str.lower)
    test_file = test_files[0]
    log.debug(f"Found the following \".test\" files: {test_files}")

    header = None
    if datafiles_with_header is True:
        header = 0
    log.info(f"Read training data from {DATA_PATH / data_file}")
    raw_training_data = pd.read_csv(DATA_PATH / data_file, header=header, skipinitialspace=True)

    log.info(f"Read test data from {DATA_PATH / test_file}")
    raw_test_data = pd.read_csv(DATA_PATH / test_file, header=header, skipinitialspace=True)

    # preprocess data
    data_processor = TabularDatasetProcessor(raw_training_data)
    training_data = data_processor.data
    test_data = data_processor.preprocess_data(raw_test_data)

    log.debug(f"Training data shape: {training_data.shape}")
    log.debug(f"Test data shape: {test_data.shape}")

    # compare number of features of training and test set -> add missing columns to the test set
    if training_data.shape[1] != test_data.shape[1]:
        log.warning(f"Number of features in the test set ({test_data.shape[1]}) differs from the training set ({training_data.shape[1]}). Adopt training features and fill new columns with 0.")
        test_data = test_data.reindex(columns=training_data.columns, fill_value=0)

    # Get dataset shapes once prior to the training to initialize helper models
    # n_samples_org = training_data.shape[0] 
    n_features = data_processor.preprocessed_feature_count

    # contains corrected cfs with features NOT encoded (still in categorical format)
    corrected_cfs = []
    cf_count = 0

    log.info("Start training process for target model")

    # epochs
    for epoch in range(epochs):
        log.info(f"Start round {epoch+1}")

        # add counterfactuals to dataset if new CFs were added
        if len(corrected_cfs) > cf_count:
            log.debug("Add approved counterfactuals to training set")
            training_data = pd.concat([training_data, data_processor.encode_features(corrected_cfs)], ignore_index=True, axis=0)
            cf_count = len(corrected_cfs)

        # split training and test data into x and y arrays
        log.debug("Split training and test data into x and y arrays")
        X_train = training_data[training_data.columns[0:-1]]
        y_train = training_data[training_data.columns[-1]]
        X_test = test_data[test_data.columns[0:-1]]
        y_test = test_data[test_data.columns[-1]]

        log.debug(f"Training data {training_data.shape} split into X {X_train.shape} and y {y_train.shape}.")
        log.debug(f"Test data {test_data.shape} split into X {X_test.shape} and y {y_test.shape}.")

        # train target model
        log.debug("Fitting target model")
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        
        with open(file=model_logfile, mode="a", encoding="utf-8") as file:
            file.write(f"##### Epoch {epoch} #####\n")
            file.write(f"mean accuracy of model: {score}\n")

        # generate counterfactuals and save in counterfactuals list
        for index in range(X_train.shape[0]):
            log.debug(f"Current instance index: {index} (training sample {index +1}).")

            # build leave one out training data set for helper models
            X_helper_train = X_train.drop(labels=index, axis=0)
            y_helper_train = y_train.drop(labels=index, axis=0)

            # initialize helper models with standard parameters
            helper_models = init_helper_models(n_features=n_features, seed=SEED)

            # train helper models with all training and test instances except the current one
            helper_accuracies = []  # dim: 1 x number_helper_models
            log.debug("Start training of the helper models")

            for hm in helper_models:
                hm.fit(X_helper_train, y_helper_train)
                accuracy = hm.score(X_test, y_test)
                helper_accuracies.append(accuracy)

            log.debug(f"Finished training of the helper models: {helper_accuracies}")

            # generate current instance (only X_data, no y/label)
            current_instance = X_train.iloc[[index]]

            # set up explainer
            dice_data = dice_ml.Data(dataframe=training_data, continuous_features=data_processor.numerical_features, outcome_name=data_processor.target_name)
            dice_model = dice_ml.Model(model=model, backend="sklearn")
            explainer = dice_ml.Dice(dice_data, dice_model, method="random")

            # produce counterfactual (input has to be without the target/label)
            cf = explainer.generate_counterfactuals(current_instance, total_CFs=1, desired_class="opposite", random_seed=SEED)
            cf.visualize_as_list()
            counterfactual = cf.cf_examples_list[0].final_cfs_df

            # produce prediction for counterfactual example
            helper_preds = []        # dim: 1 x number_helper_models
            helper_proba_preds = []  # dim: number_helper_models x number_classes
            
            for hm in helper_models:
                pred = hm.predict(counterfactual.iloc[:,:-1])
                helper_preds.append(pred[0])
                proba_pred = hm.predict_proba(counterfactual.iloc[:,:-1])
                helper_proba_preds.append(proba_pred[0])

            log.debug(f"Sum of helper model accuracies: {np.sum(helper_accuracies)}.")
            log.debug(f"Helper model predictions for current instance: {helper_preds}")
            log.debug(f"Helper model probabilities for current instance: {helper_proba_preds}")

            # calculate uncertainty of helper_models
            weighted_bagging_score = np.dot(helper_accuracies, helper_preds)/np.sum(helper_accuracies)    # scalar value between 0 and highest class number/encoding
            weighted_bagging_pred = round(weighted_bagging_score)
            weighted_probabs = np.dot(helper_accuracies, helper_proba_preds)/np.sum(helper_accuracies)    # dim: 1 x number_classes; values between 0 and highest class number/encoding
            weighted_probabs_pred = np.argmax(weighted_probabs)
            log.debug(f"weighted_bagging_score: {weighted_bagging_score}")
            log.debug(f"weighted_bagging_pred: {weighted_bagging_pred}")
            log.debug(f"Weighted probabilities: {weighted_probabs}")
            # print(f"weighted_probabs_pred: {weighted_probabs_pred}")
            
            # if helper model probability is below the given threshold, present counterfactual to human expert
            if weighted_probabs[weighted_probabs_pred] < threshold:
                # present counterfactual to human expert for correction and retrieve corrected version
                cf_for_user = data_processor.inverse_transform_data_and_target(counterfactual)
                log.info(f"Found a CF for review!\n{cf_for_user.T}")
                
                cf_for_training = cf_for_user

                if automate == False:

                    # check with user if counterfactual needs to be corrected at all
                    question = [
                        inquirer.Confirm(name="needs_editing", 
                                        message="Would you like to edit the values of the instance above?",
                                        default=True)
                    ]
                    answer = inquirer.prompt(question)

                    if answer["needs_editing"] == True:
                        question_cf = []

                        # generate prompt for features
                        for feature in cf_for_user.columns[:-1]:
                            print(f"feature: {feature}")
                            if cf_for_user[feature].dtypes == "object":
                                # get list of feature values: retrieve the index of "feature" in the categorical_features list
                                # and use index to call the list of feature values from the one hot encoder
                                feature_values = data_processor.ohc_encoder.categories_[data_processor.categorical_features.index(feature)].tolist()
                                prompt = inquirer.List(name=feature, 
                                                    message=f"Choose value of {feature}. Old value: {cf_for_user.iloc[0][feature]}",
                                                    choices=feature_values)
                                question_cf.append(prompt)
                            elif cf_for_user[feature].dtypes == "float": # all numerical features will be continous after cf generation
                                prompt = inquirer.Text(name=feature,
                                                    message=f"Enter value for {feature}. Old value: {cf_for_user.iloc[0][feature]}",
                                                    validate=validate_number) 
                                question_cf.append(prompt)     

                        # generate prompt for target
                        target = cf_for_user.columns[-1]
                        target_prompt = inquirer.List(name=target,
                                                message=f"Choose value of {target}. Old value: {cf_for_user.iloc[0][target]}",
                                                choices=data_processor.label_encoder.classes_.tolist())
                        question_cf.append(target_prompt)
                            
                        answer_cf = inquirer.prompt(question_cf)
                    
                        cf_for_training = pd.DataFrame([answer_cf])

                corrected_cfs.append(cf_for_training)

    cf_f = OUTPUT_PATH / "counterfactuals"
    with open(file=cf_f, mode="w", encoding="utf-8") as file:
        for feature in data_processor.initial_features:
            file.write(feature + ",")
        file.write("\n")
        for cf in corrected_cfs:
            file.write(cf + "\n")

    log.info("Finished training the target model")


def DEV_ONLY_create_target_model(feature_count):
    hidden_layer1_size = round(feature_count*(1.3))
    hidden_layer2_size = round(hidden_layer1_size/3)
    model = MLPClassifier(hidden_layer_sizes=(hidden_layer1_size, hidden_layer2_size),
                          activation="relu", solver="adam", random_state=SEED)
    log.debug(f"target model: {model}")
    return model

def validate_number(_, current) -> Boolean:
    current = float(current)
    return isinstance(current, float)


def main():

    # TO DO: ArgParsing
    # import argparse
    # arguments: model
    # optional arguments: data_folder, counterfactual generator, rounds
    # technical optional arguments: seed, log-level

    # dev_path = Path()/"data"/"adult_income"/"small"
    dev_path = Path()/"data"/"adult_income"
    output_path = Path()/"data"/"output"

    logfile = output_path / "logfile"
    # clear logfile
    if logfile.is_file():
        open(file=logfile, mode="w").close()
    
    log.basicConfig(filename=logfile,
                    filemode="a",
                    format="%(asctime)s %(levelname)s: %(message)s", 
                    datefmt="%d/%m/%Y %H:%M",
                    level=log.DEBUG)
    log.info("Start program")
    log.debug(f"Seed: {SEED}")

    ########## DEV only ##########

    threshold = 0.7



    # manually set known value for input size of the model --> dataset specific
    model = DEV_ONLY_create_target_model(104)

    ########## DEV only ##########
    
    # train_model_with_cfs(data_path=dev_path, datafiles_with_header=True, model=model, epochs=5, threshold=threshold)
    train_model_with_cfs(data_path=dev_path, datafiles_with_header=True, model=model, epochs=5, threshold=threshold, automate=True, output_path=output_path)

    log.info("End program")


if __name__ == '__main__':
    main()
    