import numpy as np
from sklearn.model_selection import train_test_split

from classification_model.config.core import config
from classification_model.pipeline import titanic_pipe
from classification_model.processing.data_manager import load_dataset, save_pipeline

def run_training() -> None:
    print("Training!!!!!!!!!!!!!!!!!!")

    # Read Training data
    data = load_dataset(file_name=config.app_config.raw_data_file)
    print('------------- ', data.shape)

    # Divide train and test set
    X_train, X_test, y_train, y_test = train_test_split(
        data[config.model_config.features], # predictors
        data[config.model_config.target],
        test_size = config.model_config.test_size,
        # We are setting the random seed here for reproducibility
        random_state = config.model_config.random_state,
    )
    print('X_train --- ', X_train.shape)
    # Train and Fit model
    titanic_pipe.fit(X_train, y_train)

    # Persist trained model
    save_pipeline(pipeline_to_persist=titanic_pipe)


if __name__ == '__main__':
    run_training()
