from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
# from sklearn.svm import LinearSVC

# models should be for binary and multi-class classification
def init_helper_models(n_features: int, seed: int):
    """
    initializes three helper models with standard configuration: GradientBoosting, MLP, RandomForest

    Parameters
    ----------
    n_features: int
        number of features (= input layer size for MLP)

    seed: int
        for setting the random state of the helper models

    Returns
    -------
    list
        list containing the helper models
    
    """
    
    HELPER_MODELS = []

    # helper_SVM = LinearSVC(random_state=seed)  
    
    # scikit-learn doc: dual or primal optimization problem. Prefer dual=False when n_samples > n_features.
    # if n_samples>n_features:
    #     helper_SVM.dual=False

    helper_GB = GradientBoostingClassifier(random_state=seed)
    
    helper_MLP = MLPClassifier(random_state=seed, hidden_layer_sizes=(n_features))

    helper_RF = RandomForestClassifier(random_state=seed)

    HELPER_MODELS.extend([helper_GB, helper_MLP, helper_RF])
    
    return HELPER_MODELS