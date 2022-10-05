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

mode = MODE_AUTOMATE_GEN
SEED = 42

def train_model_with_cfs(data_path: Path(), datafiles_with_header: bool, model, epochs: int, threshold_hm_acc: float, output_path=Path()):

    # validate user input
    # if datafiles contain no header, give user the option to exit program
    if datafiles_with_header is False:
        log.critical("Providing a header in your data files is vital in order for the human expert to validate the counterfactual examples in an appropriate time. Please provide a header in your files.")
        log.critical("Exit program")
        exit()
    
    # validate threshold
    if isinstance(threshold_hm_acc, float):
        if not threshold_hm_acc == round(threshold_hm_acc, 3):
            log.warning(f"The provided threshold {threshold_hm_acc} will be rounded to three decimals. New threshold: {round(threshold_hm_acc, 3)}.")
        threshold_hm_acc = round (threshold_hm_acc, 3)
    else:
        log.critical(f"The provided threshold was not valid. Value: {threshold_hm_acc}. Expected value of type float.")
        log.critical("Exit program")
        exit()

    log.info(f"Number of rounds: {epochs}")

    # data paths
    DATA_PATH = Path(data_path)
    if not DATA_PATH.exists() or not DATA_PATH.is_dir():
        log.critical(f"{DATA_PATH} was provided as data path. Path is not valid. Please check your input. The program will exit now.")
        log.critical("Exit program")
        exit()
    log.debug(f"Set data path: {DATA_PATH}.")

    # path to file containing corrected counterfactuals
    CORRECTED_CF_FILE = DATA_PATH / "corrected_counterfactuals.csv"

    if mode == MODE_AUTOMATE_TRAIN:
        if not CORRECTED_CF_FILE.exists() or not CORRECTED_CF_FILE.is_file():
            log.critical(f"Corrected counterfactuals are expected to be in {CORRECTED_CF_FILE} but this is not valid. Please make sure that the counterfactuals are in a file named \"corrected_counterfactuals.csv\" in the data path you provide. The program will exit now.")
            log.critical("Exit program")
            exit()
        else:
            log.debug("Found \"corrected_counterfactuals.csv\".")

    # general output paths + creation
    if not Path(output_path).exists() or not Path(output_path).is_dir():
        log.critical(f"{Path(output_path)} was provided as output path. Path is not valid. Please check your input. The program will exit now.")
        log.critical("Exit program")
        exit()
    
    folder_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    OUTPUT_PATH = Path(output_path) / f"{DATA_PATH.name}_{folder_timestamp}_t-"
    log.debug(f"Create output directory {OUTPUT_PATH} in {Path(output_path)}.")
    OUTPUT_PATH.mkdir()    

    # create model output path 
    MODEL_OUTPUT_PATH = OUTPUT_PATH / "models"
    MODEL_OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    
    # output files paths
    cf_file = OUTPUT_PATH / "counterfactuals.csv"
    cf_proba_file = OUTPUT_PATH / "counterfactual_probabilities"
    no_cf_instances_file = OUTPUT_PATH / "instances_without_counterfactual.csv"
    model_logfile = MODEL_OUTPUT_PATH / "model_performance"

    file_list = [cf_file, cf_proba_file, no_cf_instances_file]

    # clear file content
    for file in file_list:
        if file.is_file():
            open(file=file, mode="w", encoding="utf-8").close()

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

    # preprocess data
    data_processor = TabularDatasetProcessor(raw_training_data)

    # read corrected counterfactuals from file
    corrected_cfs_from_file = None
    if mode == MODE_AUTOMATE_TRAIN:
        if CORRECTED_CF_FILE.stat().st_size == 0:
            log.critical(f"{CORRECTED_CF_FILE} is empty. Please proved counterfactuals or start program in a different mode. The program will exit now.")
            log.critical("Exit program")
            exit()
        corrected_cfs_from_file = read_corrected_cfs_from_file(path=CORRECTED_CF_FILE, data_header=data_processor.initial_features)

    # get encoded training and test data
    training_data_enc = data_processor.data
    test_data_enc = data_processor.preprocess_data(raw_test_data)

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
                                X_test=X_test, y_test=y_test, cf_count=len(corrected_cfs))

    # epochs
    for epoch in range(epochs):
        log.info(f"Start round {epoch+1} of {epochs}.")

        # set up explainer (updated every epoch because model changes, too)
        dice_data = dice_ml.Data(dataframe=training_data_enc, continuous_features=data_processor.numerical_features, outcome_name=data_processor.target_name)
        dice_model = dice_ml.Model(model=model, backend="sklearn")
        explainer = dice_ml.Dice(dice_data, dice_model, method="random")

        # generate counterfactuals and save in counterfactuals list
        for index in range(X_train.shape[0]):
            log.debug(f"Index of current instance: {index}.")
            # if index == 100:
            #     break

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
                    question = [
                        inquirer.Confirm(name="needs_editing", 
                                        message="Would you like to edit the values of the instance above?",
                                        default=True)
                    ]
                    answer = inquirer.prompt(question)

                    if answer["needs_editing"] == True:
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
                                    X_test=X_test, y_test=y_test, epoch=epoch+1, cf_count=cf_count)
    
    # write all (corrected) counterfactuals used for training to a file
    cfs_df = pd.concat(corrected_cfs, axis=0, ignore_index=True)
    cfs_df.to_csv(path_or_buf=cf_file, mode="w", header=True, index=False, encoding="utf-8")

    with open(file=cf_proba_file, mode="w", encoding="utf-8") as f:
        for p in corrected_cfs_proba:
            f.write(f"{p}\n")

    log.info("Finished training the target model")

def read_corrected_cfs_from_file(path: Path, data_header: list) -> pd.DataFrame:
    raw_data = pd.read_csv(path, header=0, skipinitialspace=True)
    if not raw_data.columns == data_header:
        log.critical(f"Header of {path} does not match dataset header. Please adapt the files. The program will exit now.")
        exit()
    return raw_data


def split_data_in_xy(dataset: pd.DataFrame):
    X_data = dataset[dataset.columns[0:-1]]
    y_data = dataset[dataset.columns[-1]]
    return X_data,y_data


def train_and_log_target_model(model, model_logfile: Path, model_output: Path(), X_train: pd.DataFrame, 
                y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series, cf_count: int, epoch=0):
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


def DEV_ONLY_create_target_model(feature_count):
    hidden_layer1_size = round(feature_count*(1.3))
    hidden_layer2_size = round(hidden_layer1_size/3)
    model = MLPClassifier(hidden_layer_sizes=(hidden_layer1_size, hidden_layer2_size),
                          activation="relu", solver="adam", random_state=SEED)
    return model


def validate_number(_, current) -> bool:
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
    output_path = Path()/"output"
    output_path.mkdir(parents=True, exist_ok=True)

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
    log.debug(f"Seed: {SEED}")

    ########## DEV only ##########

    threshold = 0.55

    # manually set known value for input size of the model --> dataset specific
    model = DEV_ONLY_create_target_model(104)

    ########## DEV only ##########


    # train_model_with_cfs(data_path=dev_path, datafiles_with_header=True, model=model, epochs=5, threshold=threshold)
    train_model_with_cfs(data_path=dev_path, datafiles_with_header=True, model=model, epochs=5, 
                threshold_hm_acc=threshold, output_path=output_path, )

    log.info("End program")


if __name__ == '__main__':
    main()
    