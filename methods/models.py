from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from methods.classifier import Classifier, SVC
import gin


@gin.configurable
def decision_tree(**kwargs):
    """

    :param kwargs:          hyperparameters to the classifier, which is being defined in configs/hyperparameters.gin
    :return:                Classifier, a custom class with functions   
    """
    return Classifier(DecisionTreeClassifier(**kwargs), 'Decision Tree')


@gin.configurable
def random_forest(**kwargs):
    return Classifier(RandomForestClassifier(**kwargs), 'Random Forest')


@gin.configurable
def svc(**kwargs):
    return SVC(svm.SVC(**kwargs), 'SVC')


@gin.configurable
def ada_boost(**kwargs):
    return Classifier(AdaBoostClassifier(**kwargs), 'Ada Boost')


@gin.configurable
def knn(**kwargs):
    return Classifier(KNeighborsClassifier(**kwargs), 'KNN')
