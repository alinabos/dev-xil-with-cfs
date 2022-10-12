from datetime import datetime
import dice_ml
from dice_ml.utils.exception import UserConfigValidationException
import inquirer
from joblib import dump
import logging as log
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.neural_network import MLPClassifier
from src.TabularDatasetProcessor import TabularDatasetProcessor
from src.helper_models import init_helper_models

MODE_AUTOMATE_GEN = "automate_gen"      # automate + generate counterfactuals that user should correct
MODE_AUTOMATE_TRAIN = "automate_train"  # automate + read corrected counterfactuals from file
MODE_INTERACTIVE = "interactive"        # interactive (counterfactuals are corrected live while training)
modes = [MODE_AUTOMATE_GEN, MODE_AUTOMATE_TRAIN, MODE_INTERACTIVE]


def train_model_with_cfs(data_path: Path(), datafiles_with_header: bool, model, epochs: int, threshold_hm_acc: float, output_path: Path, use_timestamp: bool, seed: int, mode: str):

    # validate user input
    DATA_PATH, MODEL_OUTPUT_PATH, output_file_list = process_arguments(data_path, datafiles_with_header, epochs, threshold_hm_acc, output_path, use_timestamp, mode)
    CORRECTED_CF_FILE = check_counterfactual_file(DATA_PATH, mode)
    SEED = seed

    # set file names
    cf_file = output_file_list[0]
    cf_proba_file = output_file_list[1]
    no_cf_instances_file = output_file_list[2]
    model_logfile = output_file_list[3]

    # read data from data path
    raw_training_data, raw_test_data = read_datasets_from_datapath(DATA_PATH)

    # preprocess data
    data_processor = TabularDatasetProcessor(raw_training_data)

    # read corrected counterfactuals from file
    corrected_cfs_from_file = get_corrected_cfs_from_file(file=CORRECTED_CF_FILE, mode=mode, header=data_processor.initial_features)

    # get encoded training and test data
    training_data_enc = data_processor.data
    test_data_enc = data_processor.preprocess_data(raw_test_data)

    # validate threshold against number of classes found
    validate_threshold_against_classes(threshold_hm_acc, data_processor.label_encoder.classes_)

    log.debug(f"Training data shape: {training_data_enc.shape}")
    log.debug(f"Test data shape: {test_data_enc.shape}")

    # compare number of features of training and test set -> add missing columns to the test set
    if training_data_enc.shape[1] != test_data_enc.shape[1]:
        log.warning(f"Number of features in the test set ({test_data_enc.shape[1]}) differs from the training set ({training_data_enc.shape[1]}). Adopt training features and fill new columns with 0.")
        test_data_enc = test_data_enc.reindex(columns=training_data_enc.columns, fill_value=0)

    # required for the initilization of the helper models
    n_features = data_processor.preprocessed_feature_count

    # split test data into x and y arrays
    log.debug("Split training and test data into x and y arrays")

    X_train, y_train = split_data_in_xy(training_data_enc)
    X_test, y_test = split_data_in_xy(test_data_enc)

    log.debug(f"Training data {training_data_enc.shape} split into X {X_train.shape} and y {y_train.shape}.")
    log.debug(f"Test data {test_data_enc.shape} split into X {X_test.shape} and y {y_test.shape}.")
    
    log.info("Start general training process")

    # initialize helper models with standard parameters
    helper_models = init_helper_models(n_features=n_features, seed=SEED)

    helper_accuracies = []  # dim: 1 x number_helper_models
    log.debug("Start training of the helper models")

    # train helper models once with original data and save to file
    for hm in helper_models:
        hm.fit(X_train, y_train)
        accuracy = hm.score(X_test, y_test)
        helper_accuracies.append(accuracy)
        hm_path = MODEL_OUTPUT_PATH / f"{type(hm).__name__}"
        dump(hm, hm_path)

    log.debug(f"Finished training of the helper models: {helper_accuracies}")

    # contains corrected cfs with features NOT encoded (still in categorical format)
    corrected_cfs = []
    cf_count = 0

    corrected_cfs_proba = []
    
    train_and_log_target_model(model=model, model_logfile=model_logfile, model_output=MODEL_OUTPUT_PATH, X_train=X_train, y_train=y_train, 
                                X_test=X_test, y_test=y_test, cf_count=len(corrected_cfs), mode=mode)

    # epochs
    for epoch in range(epochs):
        log.info(f"Start epoch {epoch+1} of {epochs}.")

        # set up explainer (updated every epoch because model changes, too)
        dice_data = dice_ml.Data(dataframe=training_data_enc, continuous_features=data_processor.numerical_features, outcome_name=data_processor.target_name)
        dice_model = dice_ml.Model(model=model, backend="sklearn")
        explainer = dice_ml.Dice(dice_data, dice_model, method="random")

        # generate counterfactuals and save in counterfactuals list
        for index in range(X_train.shape[0]):
            log.debug(f"Index of current instance: {index}.")
            if index == 100:
                break

            # get current instance (only X_data, no y/label)
            current_instance_enc = X_train.iloc[[index]]

            log.debug("Generate counterfactuals for current instance")

            try:
                # produce counterfactual for current instance (input has to be without the target/label)
                generated_cfs_enc = explainer.generate_counterfactuals(current_instance_enc, total_CFs=3, desired_class="opposite", random_seed=SEED)

            except UserConfigValidationException as e:
                log.warning(f"{type(e)}: No counterfactuals found for the current instance. Write instance to file {no_cf_instances_file} and continue with next training instance")
                Xy_instance = pd.concat([current_instance_enc, y_train.iloc[[index]]], axis=1)

                # if file didn't exist or was empty, print dataframe header; otherwise just the data
                if not no_cf_instances_file.exists():
                    Xy_instance.to_csv(path_or_buf=no_cf_instances_file, mode="a", header=True, index=False, encoding="utf-8")
                elif no_cf_instances_file.stat().st_size == 0:
                    Xy_instance.to_csv(path_or_buf=no_cf_instances_file, mode="a", header=True, index=False, encoding="utf-8")
                else:
                    Xy_instance.to_csv(path_or_buf=no_cf_instances_file, mode="a", header=False, index=False, encoding="utf-8")
                continue

            generated_cfs_enc.visualize_as_list()
            counterfactual_enc = None 

            # validate counterfactuals and use the first valid one
            for cf_index in range(len(generated_cfs_enc.cf_examples_list[0].final_cfs_df)):
                try:
                    cf_to_test = generated_cfs_enc.cf_examples_list[0].final_cfs_df.iloc[[cf_index]]
                    # test if inverse_transform produces a valid data instance
                    data_processor.inverse_transform_data_and_target(cf_to_test)
                    log.debug("Found valid counterfactual")
                    counterfactual_enc = cf_to_test
                    break
                except ValueError:
                    log.debug(f"Counterfactual at index {cf_index} was not valid. Next counterfactual in the list is tested.")
            
            # if no valid counterfactual was found, skip to next training instance
            if type(counterfactual_enc) != pd.DataFrame:
                log.critical("No valid counterfactual found. Continue with next training sample.")
                continue

            # produce prediction for counterfactual example
            helper_preds = []        # dim: 1 x number_helper_models
            helper_proba_preds = []  # dim: number_helper_models x number_classes
            
            for hm in helper_models:
                pred = hm.predict(counterfactual_enc.iloc[:,:-1])   # remove target column for prediction
                helper_preds.append(pred[0])
                proba_pred = hm.predict_proba(counterfactual_enc.iloc[:,:-1])
                helper_proba_preds.append(proba_pred[0])

            log.debug(f"Helper model predictions for counterfactual: {helper_preds}")
            log.debug(f"Helper model probabilities for counterfactual: {helper_proba_preds}")

            # calculate uncertainty of helper_models
            weighted_bagging_score = np.dot(helper_accuracies, helper_preds)/np.sum(helper_accuracies)    # scalar value between 0 and highest class number/encoding
            weighted_probabs = np.dot(helper_accuracies, helper_proba_preds)/np.sum(helper_accuracies)    # dim: 1 x number_classes; values between 0 and highest class number/encoding
            weighted_probabs_pred = np.argmax(weighted_probabs)
            
            log.debug(f"weighted_bagging_score: {weighted_bagging_score}")
            log.debug(f"Weighted_probabs_pred: {weighted_probabs_pred}. Weighted probabilities: {weighted_probabs}")
            
            # if helper model probability is below the given threshold, present counterfactual to human expert
            if weighted_probabs[weighted_probabs_pred] < threshold_hm_acc:

                # present counterfactual to human expert for correction and retrieve corrected version
                counterfactual = data_processor.inverse_transform_data_and_target(counterfactual_enc)
                log.info(f"Found a CF for review!\n{counterfactual.T}")

                # get corrected counterfactual depending on selected mode
                # MODE_AUTOMATE_GEN (standard)
                cf_for_training = counterfactual

                # MODE_INTERACTIVE: generate user prompts
                if mode == MODE_INTERACTIVE:
                    # check with user if counterfactual needs to be corrected at all
                    question_edit = [
                        inquirer.Confirm(name="needs_editing", 
                                        message="Would you like to edit the values of the instance above?",
                                        default=True)
                    ]
                    answer_edit = inquirer.prompt(question_edit)

                    if answer_edit["needs_editing"] == True:
                        question_cf = []

                        # generate prompt for features
                        for feature in counterfactual.columns[:-1]:
                            print(f"feature: {feature}")
                            if counterfactual[feature].dtypes == "object":
                                # get list of feature values: retrieve the index of "feature" in the categorical_features list
                                # and use index to call the list of feature values from the one hot encoder
                                feature_values = data_processor.ohc_encoder.categories_[data_processor.categorical_features.index(feature)].tolist()
                                prompt = inquirer.List(name=feature, 
                                                    message=f"Choose value of {feature}. Old value: {counterfactual.iloc[0][feature]}",
                                                    choices=feature_values)
                                question_cf.append(prompt)
                            elif counterfactual[feature].dtypes == "float": # all numerical features will be continous after cf generation
                                prompt = inquirer.Text(name=feature,
                                                    message=f"Enter value for {feature}. Old value: {counterfactual.iloc[0][feature]}",
                                                    validate=validate_number) 
                                question_cf.append(prompt)     

                        # generate prompt for target
                        target = counterfactual.columns[-1]
                        target_prompt = inquirer.List(name=target,
                                                message=f"Choose value of {target}. Old value: {counterfactual.iloc[0][target]}",
                                                choices=data_processor.label_encoder.classes_.tolist())
                        question_cf.append(target_prompt)
                        
                        # prompt user for input and save answer in dataframe
                        answer_cf = inquirer.prompt(question_cf)
                        cf_for_training = pd.DataFrame([answer_cf])

                # MODE_AUTOMATE_TRAIN: get corrected counterfactual from file
                elif mode == MODE_AUTOMATE_TRAIN:
                    corrected_cfs_from_file.reset_index(drop=True)
                    cf_for_training = corrected_cfs_from_file.iloc[[0]]
                    corrected_cfs_from_file.drop(0, axis=0, inplace=True)

                # append corrected counterfactual to counterfactual list
                corrected_cfs.append(cf_for_training)
                corrected_cfs_proba.append(weighted_probabs[weighted_probabs_pred])

        # add new counterfactuals to dataset if new CFs were created
        if len(corrected_cfs) > cf_count:
            log.debug("Add approved new counterfactuals to training set")

            # build one dataframe from list of one-row dataframes and apply encoding
            new_cfs_df = pd.concat(corrected_cfs[cf_count:], axis=0, ignore_index=True)
            new_cfs_df.reset_index(drop=True, inplace=True)
            new_cfs_df_enc = data_processor.transform_features_and_target(new_cfs_df)

            # split counterfactual data in X and y data
            cfs_X_data, cfs_y_data = split_data_in_xy(new_cfs_df_enc)

            # append counterfactual data to training set
            X_train = pd.concat([X_train, cfs_X_data], ignore_index=True, axis=0)
            y_train = pd.concat([y_train, cfs_y_data], ignore_index=True, axis=0)

            cf_diff = len(new_cfs_df.index)
            log.info(f"{cf_diff} new counterfactuals were added to the training set.")

            cf_count = len(corrected_cfs)

        # train model with new dataset
        train_and_log_target_model(model=model, model_logfile=model_logfile, model_output=MODEL_OUTPUT_PATH, X_train=X_train, y_train=y_train, 
                                    X_test=X_test, y_test=y_test, epoch=epoch+1, cf_count=cf_count, mode=mode)
    
    # write all (corrected) counterfactuals used for training to a file
    cfs_df = pd.concat(corrected_cfs, axis=0, ignore_index=True)
    cfs_df.to_csv(path_or_buf=cf_file, mode="w", header=True, index=False, encoding="utf-8")

    with open(file=cf_proba_file, mode="w", encoding="utf-8") as f:
        for p in corrected_cfs_proba:
            f.write(f"{p}\n")

    log.info("Finished training the target model")


def validate_number(_, current) -> bool:
    current = float(current)
    return isinstance(current, float)


def train_and_log_target_model(model, model_logfile: Path, model_output: Path(), X_train: pd.DataFrame, 
                y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series, cf_count: int, mode: str, epoch=0):
    # train target model
    log.debug("Fitting target model")
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)

    model_timestamp = datetime.now().strftime("%d.%m.%Y %H:%M:%S")

    # write target model performance to file        
    with open(file=model_logfile, mode="a", encoding="utf-8") as file:
        if epoch == 0:
            file.write(f"{model_timestamp}\n##### Initial fit #####\n{str(model)}")
        else:
            file.write(f"\n\n{model_timestamp}\n##### Epoch {epoch} #####")

        file.write(f"\nmean accuracy of model: {score}")

        if mode == MODE_AUTOMATE_GEN:
            file.write(f"\nCurrent number of counterfactual to correct (and also used for training): {cf_count}")
        else:
            file.write(f"\nCurrent number of counterfactual used for training: {cf_count}")

    # save model to disk
    out_filename = model_output / f"model_{epoch}"
    dump(model, out_filename)
    

def split_data_in_xy(dataset: pd.DataFrame):
    X_data = dataset[dataset.columns[0:-1]]
    y_data = dataset[dataset.columns[-1]]
    return X_data,y_data


def validate_threshold_against_classes(threshold: float, classes: np.ndarray):
    if threshold < 1/len(classes):
        log.critical(f"Provided threshold ({threshold}) is not valid. It has to be larger than 1/(number of classes) - meaning larger than pure chance. Number of classes found: {len(classes)} ({classes}). The program will exit now.")
        log.critical("Exit program")
        exit()


def get_corrected_cfs_from_file(file: Path(), mode: str, header: list):
    corrected_cfs_from_file = None
    if mode == MODE_AUTOMATE_TRAIN:
        if file.stat().st_size == 0:
            log.critical(f"{file} is empty. Please proved counterfactuals or start program in a different mode. The program will exit now.")
            log.critical("Exit program")
            exit()

        corrected_cfs_from_file = pd.read_csv(file, header=0, skipinitialspace=True)
        if not corrected_cfs_from_file.columns == header:
            log.critical(f"Header of {file} does not match dataset header. Please adapt the files. The program will exit now.")
            exit()
    return corrected_cfs_from_file


def read_datasets_from_datapath(DATA_PATH):
    # read dataset from data_path (assuming ".data" and ".test" as file endings)
    data_files = sorted([str(file.name) for file in Path(DATA_PATH).glob("*.data")], key=str.lower)
    data_file = data_files[0]
    log.debug(f"Found the following \".data\" files: {data_files}")

    test_files = sorted([str(file.name) for file in Path(DATA_PATH).glob("*.test")], key=str.lower)
    test_file = test_files[0]
    log.debug(f"Found the following \".test\" files: {test_files}")

    # treat first line of files as header and read data from files
    header = 0
    log.debug(f"Read training data from {DATA_PATH / data_file}")
    raw_training_data = pd.read_csv(DATA_PATH / data_file, header=header, skipinitialspace=True)

    log.debug(f"Read test data from {DATA_PATH / test_file}")
    raw_test_data = pd.read_csv(DATA_PATH / test_file, header=header, skipinitialspace=True)
    return raw_training_data,raw_test_data


def check_counterfactual_file(data_path: Path, mode: str) -> Path:

    # path to file containing corrected counterfactuals
    corrected_cf_file = data_path / "corrected_counterfactuals.csv"

    if mode == MODE_AUTOMATE_TRAIN:
        if not corrected_cf_file.exists() or not corrected_cf_file.is_file():
            log.critical(f"Corrected counterfactuals are expected to be in {corrected_cf_file} but this is not valid. Please make sure that the counterfactuals are in a file named \"corrected_counterfactuals.csv\" in the data path you provide. The program will exit now.")
            log.critical("Exit program")
            exit()
        else:
            log.debug("Found \"corrected_counterfactuals.csv\".")

    return corrected_cf_file


def process_arguments(data_path, datafiles_with_header, epochs, threshold_hm_acc, output_path, use_timestamp, mode) -> tuple[Path, Path, list]:

    # validate data paths
    data_path = Path(data_path)
    if not data_path.exists() or not data_path.is_dir():
        log.critical(f"{data_path} was provided as data path. Path is not valid. Please check your input. The program will exit now.")
        log.critical("Exit program")
        exit()
    log.debug(f"Set data path: {data_path}.")

    # if datafiles contain no header, give user the option to exit program
    if datafiles_with_header is False:
        log.critical("Providing a header in your data files is vital in order for the human expert to validate the counterfactual examples in an appropriate time. Please provide a header in your files. Program will exit now.")
        log.critical("Exit program")
        exit()

    log.info(f"Number of epochs: {epochs}")

    # validate threshold
    if threshold_hm_acc > 1 or threshold_hm_acc < 0:
        log.critical(f"Threshold must have a value between 0 and 1. Given value: {threshold_hm_acc}. Please provide a valid threshold. Program will exit now.")
        log.critical("Exit program")
        exit()

    # validate output path
    output_path = Path(output_path)
    if not output_path.exists() or not output_path.is_dir():
        log.critical(f"{output_path} does not exist.")

        # check with user if output path should be created
        question_op = [
        inquirer.Confirm(name="create_output_path", 
                        message=f"Would you like to create the output path: {output_path}?",
                        default=True)
        ]
        answer_op = inquirer.prompt(question_op)

        # if yes, create directory, else exit program
        if answer_op["create_output_path"] == True:
            output_path.mkdir(parents=True, exist_ok=True)
        else:
            log.critical("Output path will not be created. Program will exit now.")
            log.critical("Exit program")
            exit()
    log.debug(f"Set output path: {output_path}.")

    # create output folder for dataset and timestamp
    if use_timestamp == True:
        folder_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        output_folder = Path(output_path) / f"{data_path.name}_{folder_timestamp}"
        log.debug(f"Create output directory {output_folder} in {Path(output_path)}.")
        output_folder.mkdir()  
    else:
        output_folder = Path(output_path) / f"{data_path.name}"
        if output_folder.is_dir():
            log.critical(f"Output folder {output_folder} already exists. Files in this directory will be overwritten if you continue.")

            # check with user if output path should be created
            question_overwrite = [
            inquirer.Confirm(name="overwrite_files", 
                            message=f"Output folder {output_folder} already exists. Files in this directory will be overwritten if you continue. Continue?",
                            default=False)
            ]
            answer_overwrite = inquirer.prompt(question_overwrite)

            if answer_overwrite["overwrite_files"] == False:
                log.critical("User input: Do not continue. Program will exit now.")
                log.critical("Exit program")
                exit()
  
    # create model output path 
    model_output_path = output_folder / "models"
    model_output_path.mkdir(parents=True, exist_ok=True)

    # output files paths
    cf_file = output_folder / "counterfactuals.csv"
    cf_proba_file = output_folder / "counterfactual_probabilities"
    no_cf_instances_file = output_folder / "instances_without_counterfactual.csv"
    model_logfile = model_output_path / "model_performance"

    file_list = [cf_file, cf_proba_file, no_cf_instances_file]

    # clear file content
    for file in file_list:
        if file.is_file():
            open(file=file, mode="w", encoding="utf-8").close()

    file_list.append(model_logfile)

    # validate mode
    if mode not in modes:
        log.critical(f"Mode ({mode}) is not valid. Program will exit now.")
        log.critical("Exit program")
        exit()

    return data_path, model_output_path, file_list


def DEV_ONLY_create_target_model(feature_count, seed):
    hidden_layer1_size = round(feature_count*(1.3))
    hidden_layer2_size = round(hidden_layer1_size/3)
    model = MLPClassifier(hidden_layer_sizes=(hidden_layer1_size, hidden_layer2_size),
                          activation="relu", solver="adam", random_state=seed)
    return model


def main():

    # TO DO: ArgParsing
    # import argparse
    # arguments: model
    # optional arguments: data_folder, counterfactual generator, rounds
    # technical optional arguments: seed, log-level

    ###########################

    # user arguments
    mode = MODE_AUTOMATE_GEN
    dev_path = Path()/"data"/"adult_income"
    file_with_header = True
    seed = 42
    model = DEV_ONLY_create_target_model(104, seed)   # manually set known value for input size of the model --> dataset specific
    epochs = 5
    threshold = 0.70
    output_path = Path()/"output"
    use_timestamp = False
    mode = MODE_AUTOMATE_GEN

    ###########################


    logfile = output_path / "logfile"

    # clear logfile
    if logfile.is_file():
        open(file=logfile, mode="w", encoding="utf-8").close()
    
    log.basicConfig(filename=logfile,
                    filemode="a",
                    format="%(asctime)s %(levelname)s: %(message)s", 
                    datefmt="%d/%m/%Y %H:%M",
                    level=log.DEBUG)


    log.info("Start program")

    train_model_with_cfs(mode=mode, data_path=dev_path, datafiles_with_header=file_with_header, model=model, epochs=epochs, 
                threshold_hm_acc=threshold, output_path=output_path, use_timestamp=use_timestamp, seed=seed)

    log.info("End program")


if __name__ == '__main__':
    main()
    