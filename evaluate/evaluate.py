from sklearn import metrics
import matplotlib.pyplot as plt


"""
:param y_true:      1d binary array, the true output values
:param methods:     dictionary[str, 1d binary array], the predicted output values for a method
:param filename:    string, name of file to save the evaluation, if empty nothing will be saved

The functions iterate through all methods in the dictionary and for each finds the confusion matrix and calculate the 
accuracy, misclassification, precision, recall, specificity, false positive rate, false negative rate and f1. If the 
filename is given it will save the evaluation in the file, regardless it will print the evaluation to terminal.
"""
def sklearn_print_evaluation(y_true, methods, filename=""):
    output = ""
    print(methods['DNN'])
    for key in methods.keys():
        tn, fp, fn, tp = metrics.confusion_matrix(y_true, methods[key]).ravel()
        output += f'------ {key} ------\n'
        output += '\tConfusion matrix:\n'
        output += '\t\t\t\t| Pred NO\t\t| Pred YES\n'
        output += '\t------------+---------------+-----------\n'
        output += f'\tActual NO\t| tn={tn}\t\t| fp={fp}\n'
        output += f'\tActual YES\t| fn={fn}\t\t| tp={tp}\n'
        output += '\t------------+---------------+-----------\n'
        output += f'\tAccuracy: \t\t\t\t{round(accuracy(tp, tn, (tn+fp+fn+tp)), 3)}\n'
        output += f'\tMisclassification: \t\t{round(misclassification(tp, tn, (tn+fp+fn+tp)), 3)}\n'
        output += f'\tPrecision: \t\t\t\t{round(precision(tp, fp), 3)}\n'
        output += f'\tRecall: \t\t\t\t{round(recall(tp, fn), 3)}\n'
        output += f'\tSpecificity: \t\t\t{round(specificity(tn, fp), 3)}\n'
        output += f'\tFalse positive rate: \t{round(false_positive_rate(fp, tn), 3)}\n'
        output += f'\tFalse negative rate: \t{round(false_negative_rate(fn, tp), 3)}\n'
        output += f'\tF1: \t\t\t\t\t{round(f1(tp, fp, fn), 3)}\n'
        output += '\n\n'
    if len(filename) > 0:
        f = open(filename, "w")
        f.write(output)
        f.close()
    print(output)


def sklearn_auc(y_true, methods):
    for key in methods.keys():
        auc = metrics.roc_auc_score(y_true, methods[key])
        fpr, tpr, thresholds = metrics.roc_curve(y_true, methods[key])
        plt.plot(fpr, tpr, marker='.')
    plt.plot([0, 1], [0, 1], linestyle='--')  # plt no skill
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    labels = list(methods.keys())
    labels.append('No skill')
    plt.legend(labels)
    plt.title(f'{key}: AUC = {round(auc * 100, 2)}')
    plt.show()


"""
:param tp:      int, number of true positives 
:param tn:      int, number of true negatives
:param total:   int, total number of cases

:return :       float, the accuracy

Calculating and returning the accuracy for the given case.
"""
def accuracy(tp, tn, total):
    return (tp + tn)/total


"""
:param tp:      int, number of true positives 
:param tn:      int, number of true negatives
:param total:   int, total number of cases

:return :       float, the misclassification

Calculating and returning the misclassification, the percentage of cases that was wrongly classified
"""
def misclassification(tp, tn, total):
    return 1 - accuracy(tp, tn, total)


"""
:param tp:      int, number of true positives 
:param fp:      int, number of false positives

:return :       float, the precision

Calculating and returning the precision, out of total predicted true,the percentage of how often the model predict 
accurate.
"""
def precision(tp, fp):
    return tp/(tp + fp)


"""
:param tp:      int, number of true positives 
:param fn:      int, number of false negatives

:return :       float, the recall

Calculating and returning the recall (true positive rate), number of items correctly identified as positive out of 
total true positives
"""
def recall(tp, fn):
    return tp/(tp + fn)


"""
:param tn:      int, number of true negatives 
:param fp:      int, number of false positives

:return :       float, the specificity

Calculating and returning the specificity (true negative rate), number of items correctly identified as negative out 
of total negatives.
as positive.
"""
def specificity(tn, fp):
    return tn/(tn + fp)


"""
:param fp:      int, number of false positives 
:param tn:      int, number of true negatives

:return :       float, the false positive rate

Calculating and returning the fpr, the fraction of all negative instances that the classifier incorrectly identifies 
as positive.
"""
def false_positive_rate(fp, tn):
    return fp/(fp + tn)


"""
:param fn:      int, number of false negatives 
:param pn:      int, number of true positives

:return :       float, the false negative rate

Calculating and returning the fnr, number of items wrongly identified as negative out of total true positives.
"""
def false_negative_rate(fn, tp):
    return fn/(fn + tp)


"""
:param tp:      int, number of true positives
:param fp:      int, number of false positives 
:param fn:      int, number of false negatives

:return :       float, f1

Calculating and returning the fpr, a score that combines precision-recall into a single number.
"""
def f1(tp, fp, fn):
    p = precision(tp, fp)
    r = recall(tp, fn)
    return (2 * r * p)/(r + p)
