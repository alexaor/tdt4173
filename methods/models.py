from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from methods.classifier import Classifier
import gin


@gin.configurable
def decision_tree(**kwargs) -> Classifier:
    """
    Method for returning a Decision Tree classifier

    Parameters
    ----------
    **kwargs
        Keyword arguments are used to inject hyperparameters from gin the gin configuration file

    Returns
    -------
    classifier
        A Decision Tree Classifier instance of the generic Classifier <methods.classifier.Classifier>
    """
    return Classifier(DecisionTreeClassifier, 'Decision Tree', **kwargs)


@gin.configurable
def random_forest(**kwargs) -> Classifier:
    """
    Method for returning a Random Forest classifier

    Parameters
    ----------
    **kwargs
        Keyword arguments are used to inject hyperparameters from gin the gin configuration file

    Returns
    -------
    classifier
        A Random Forest Classifier instance of the generic Classifier <methods.classifier.Classifier>
    """

    return Classifier(RandomForestClassifier, 'Random Forest', **kwargs)


@gin.configurable
def ada_boost(**kwargs) -> Classifier:
    """
    Method for returning an Adaptive Boosting classifier

    Parameters
    ----------
    **kwargs
        Keyword arguments are used to inject hyperparameters from gin the gin configuration file

    Returns
    -------
    classifier
        An Adaptive Boosting Classifier instance of the generic Classifier <methods.classifier.Classifier>
    """
    kwargs['base_estimator'] = DecisionTreeClassifier(max_depth=2, max_features='log2')
    return Classifier(AdaBoostClassifier, 'Ada Boost', **kwargs)
