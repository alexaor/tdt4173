from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from methods.classifier import Classifier
from gin import configurable
from methods.dnn import DNN


class Models:

    @classmethod
    @configurable
    def decision_tree(cls, **kwargs) -> Classifier:
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

    @classmethod
    @configurable
    def ada_boost(cls, **kwargs) -> Classifier:
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
        kwargs['base_estimator'] = DecisionTreeClassifier(max_depth=1)#, class_weight={0: 1, 1: 3})
        return Classifier(AdaBoostClassifier, 'Ada Boost', **kwargs)

    @classmethod
    @configurable
    def random_forest(cls, **kwargs) -> Classifier:
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

    @classmethod
    @configurable
    def dnn(cls, input_shape, initial_bias, **kwargs) -> DNN:
        """
        Method for returning a Random Forest classifier

        Parameters
        ----------
        **kwargs
            Keyword arguments are used to inject hyperparameters from gin the gin configuration file

        Returns
        -------
        DNN
            A dnn instance of the generic Classifier <methods.classifier.Classifier>
        """
        # TODO
        return DNN(input_shape=input_shape, initial_bias=initial_bias, **kwargs)
