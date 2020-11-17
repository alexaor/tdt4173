from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
import evaluate.utils as utils
from tensorflow_addons.metrics import CohenKappa


"""
:param y_true:              1d binary array, the true output values
:param methods:             dict[str, 1d binary array], the predicted output values for a method
:param filename:            string, name of file to save the evaluation, if empty nothing will be saved
:param dnn_conf_matrix:     list, the values in a confusion matrix, sorted by: tn, fp, fn, tp

The functions iterate through all methods in the dictionary and for each finds the confusion matrix and calculate the 
accuracy, misclassification, precision, recall, specificity, false positive rate, false negative rate and f1. If the 
filename is given it will save the evaluation in the file, regardless it will print the evaluation to terminal.
"""
def print_evaluation(y_true, methods, filename="", dnn_conf_matrix=None):
    output = ""
    if dnn_conf_matrix is not None:
        kappa = CohenKappa(num_classes=2)
        kappa.update_state(y_true, methods['DNN'])
        methods['DNN'] = dnn_conf_matrix
    for key in methods.keys():
        if key == 'DNN':
            tn = dnn_conf_matrix[0]; fp = dnn_conf_matrix[1]; fn = dnn_conf_matrix[2]; tp = dnn_conf_matrix[3]
        else:
            tn, fp, fn, tp = metrics.confusion_matrix(y_true, methods[key]).ravel()
        output += f'------ {key} ------\n'
        output += '\tConfusion matrix:\n'
        output += '\t\t\t\t| Pred NO\t\t| Pred YES\n'
        output += '\t------------+---------------+-----------\n'
        output += f'\tActual NO\t| tn={tn}\t\t| fp={fp}\n'
        output += f'\tActual YES\t| fn={fn}\t\t| tp={tp}\n'
        output += '\t------------+---------------+-----------\n'
        output += f'\tAccuracy: \t\t\t\t{round(accuracy(tp, tn, (tn+fp+fn+tp)), 3)}\n'
        if key != 'DNN':
            output += f'\tCohen kappa: \t\t\t{round(metrics.cohen_kappa_score(methods[key], y_true), 3)}\n'
        else:
            output += f'\tCohen kappa: \t\t\t{round(kappa.result().numpy(), 3)}\n'
        output += f'\tMisclassification: \t\t{round(misclassification(tp, tn, (tn+fp+fn+tp)), 3)}\n'
        output += f'\tPrecision: \t\t\t\t{round(precision(tp, fp), 3)}\n'
        output += f'\tRecall: \t\t\t\t{round(recall(tp, fn), 3)}\n'
        output += f'\tSpecificity: \t\t\t{round(specificity(tn, fp), 3)}\n'
        output += f'\tFalse positive rate: \t{round(false_positive_rate(fp, tn), 3)}\n'
        output += f'\tFalse negative rate: \t{round(false_negative_rate(fn, tp), 3)}\n'
        output += f'\tF1: \t\t\t\t\t{round(f1(tp, fp, fn), 3)}\n'
        output += '\n\n'
    if len(filename) > 0:
        file_path = utils.save_text(filename, output)
        print(f"Saved all evaluations in the file: '{file_path}'")

    print(output)


"""
:param y_true:              1d binary array, the true output values
:param methods:             dict[str, 1d binary array], the predicted output values for a method
:param filename:            string, name of file to save the evaluation, if empty plot will only be shown not saved

Creates a ROC curve with the given inputs, and calculate the AUC for each methods. If no filename is given the plot
will not be saved, only shown.
"""
def plot_roc_auc(y_true, methods, filename):
    auc = []
    fig, ax = plt.subplots()
    for key in methods.keys():
        fpr, tpr, _ = metrics.roc_curve(y_true, methods[key])
        auc.append(metrics.auc(fpr, tpr))
        ax.plot(fpr, tpr)
    ax.plot([0, 1], [0, 1], linestyle='--')  # plt no skill
    ax.set_xlabel('False positive rate')
    ax.set_ylabel('True positive rate')
    labels = [f'{list(methods.keys())[i]}, AUC: {round(auc[i], 2)}' for i in range(len(auc))]
    labels.append('No skill')
    fig.legend(labels, loc=7, bbox_to_anchor=(0.9, 0.3))
    ax.set_title('ROC curve')
    ax.grid(True)
    # Save plot
    file_path = utils.save_plot(fig, filename)
    print(f"Saved ROC plot at: '{file_path}'")


def plot_precision_recall(y_true, methods, filename):
    fig, ax = plt.subplots()
    for model in methods.keys():
        ap = metrics.average_precision_score(y_true, methods[model])
        pre, rec, _ = metrics.precision_recall_curve(y_true, methods[model])
        ax.plot(pre, rec, label=f'{model}, AP = {round(ap, 2)}')
    ax.set_ylabel('Precision')
    ax.set_xlabel('Recall')
    ax.set_title('Precision Recall Curve')
    ax.grid(True)
    ax.legend(loc='best')
    # Save plot
    file_path = utils.save_plot(fig, filename)
    print(f"Saved Precision Recall curve at: '{file_path}'")


"""
:param y_true:              1d binary array, the true output values
:param methods:             dictionary[str, 1d binary array], the predicted output values for a method
:param dirname:             str, name of the directory the plots will be saved in, if empty plots will only be shown
:param dnn_conf_matrix:     list, the values in a confusion matrix, sorted by: tn, fp, fn, tp

Makes a separate plot for each evaluation method, where all the methods in the dictionary 'methods' are being
compared to one another. The plots are either saved in the directory with given name, or just shown to the user.
"""
def plot_evaluation_result(y_true, methods, dirname, dnn_conf_matrix=None):
    if len(dirname) > 0:
        utils.make_plot_dir(dirname)
    evaluations = get_all_evaluations(y_true, methods, dnn_conf_matrix)
    width = 0.35

    # Plotting
    for key in evaluations.keys():
        labels = list(evaluations[key].keys())
        labels_loc = np.arange(len(labels))
        values = [value * 100 for value in evaluations[key].values()]
        fig, ax = plt.subplots()
        reacts = ax.bar(labels_loc + width, values, width)
        ax.set_title(key)
        ax.set_yticks([y*10 for y in range(1, 11)])
        ax.set_yticklabels([f'{y*10} %' for y in range(1, 11)])
        ax.set_xticks(labels_loc+width)
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 105)

        # Add the values on top of the bars
        for react in reacts:
            height = react.get_height()
            ax.annotate(round(height, 2),
                         xy=(react.get_x() + width/2, height),
                         xytext=(0, 3),
                         textcoords="offset points",
                         ha='center', va='bottom')

        # Save plot
        plot_dir = utils.save_plot(fig, f'{key}.png', dirname)
        print(f"Saved all evaluations plot in directory: '{plot_dir}'")


"""
:param y_true:              1d binary array, the true output values
:param methods:             dictionary[str, 1d binary array], the predicted output values for a method
:param evallist:            list[str], list of evaluations method that the methods shall be compared against each other
:param filename:            str, name of the file the plot will be saved as, if empty the plot will not be saved only shown
:param dnn_conf_matrix:     list, the values in a confusion matrix, sorted by: tn, fp, fn, tp

Compares the methods in the methods dictionary with the wanted evaluations method. Either saves the plot, or shows it 
to the user. 
"""
def plot_comparison(y_true, methods, evallist, filename, dnn_conf_matrix=None):
    evallist = [evaluation.lower() for evaluation in evallist]
    all_evaluations = get_all_evaluations(y_true, methods, dnn_conf_matrix)
    evaluations = {}
    width = 0.55

    # make a new dictionary which has method as keys and its values are dictionaries with evaluation method as keys
    for method in methods:
        evaluations[method] = {}
        for evaluation in list(all_evaluations.keys()):
            if evaluation.lower() in evallist:
                evaluations[method][evaluation] = all_evaluations[evaluation][method]

    # Plotting
    labels_loc = np.arange(len(evallist))
    numb_methods = len(methods.keys())
    fig, ax = plt.subplots()
    fig.set_figheight(2*numb_methods)
    fig.set_figwidth(3*numb_methods + len(evallist))
    i = 0
    for method in evaluations.keys():
        values = [value * 100 for value in evaluations[method].values()]
        reacts = ax.bar(labels_loc + i*width/numb_methods, values, width/numb_methods, label=method)
        labels = list(evaluations[method].keys())
        for react in reacts:
            height = react.get_height()
            ax.annotate(round(height, 2),
                         xy=(react.get_x() + react.get_width()/2, height),
                         xytext=(0, 3),
                         textcoords="offset points",
                         ha='center', va='bottom')
        i += 1
    ax.set_title('Comparison of  methods')
    ax.set_yticks([y * 10 for y in range(1, 11)])
    ax.set_yticklabels([f'{y * 10} %' for y in range(1, 11)])
    ax.set_xticks(labels_loc + width/numb_methods)
    ax.set_xticklabels(labels)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    fig.tight_layout()
    ax.set_ylim(0, 110)
    # Save or show plots
    plot_dir = utils.save_plot(fig, filename)
    print(f"Saved the comparison plot in directory: '{plot_dir}/{filename}'")


"""
:param y_true:              1d binary array, the true output values
:param methods:             dict[str, 1d binary array], the predicted output values for a method
:param dnn_conf_matrix:     list, the values in a confusion matrix, sorted by: tn, fp, fn, tp

:return evaluation:     dict[str, dict[str, float]], the results for the different evaluations methods

Creates a new dictionary where the evaluations method is the key, the value for the keys are a new dictionary, where 
the keys are the method and the values are the result for the given evaluation method. This dictionary is returned.
"""
def get_all_evaluations(y_true, methods, dnn_conf_matrix):
    if dnn_conf_matrix is not None:
        kappa = CohenKappa(num_classes=2)
        kappa.update_state(y_true, methods['DNN'])
        methods['DNN'] = dnn_conf_matrix
    evaluation = {'Accuracy': {},
                  'Misclassification': {},
                  'Precision': {},
                  'Recall': {},
                  'Specificity': {},
                  'False positive rate': {},
                  'False negative rate': {},
                  'F1': {},
                  'Cohen kappa': {}}
    for key in methods.keys():
        if key == 'DNN':
            tn = dnn_conf_matrix[0]; fp = dnn_conf_matrix[1]; fn = dnn_conf_matrix[2]; tp = dnn_conf_matrix[3]
        else:
            tn, fp, fn, tp = metrics.confusion_matrix(y_true, methods[key]).ravel()
        evaluation['Accuracy'][key] = accuracy(tp, tn, tn + fp + fn + tp)
        evaluation['Misclassification'][key] = misclassification(tp, tn, tn + fp + fn + tp)
        evaluation['Precision'][key] = precision(tp, fp)
        evaluation['Recall'][key] = recall(tp, fn)
        evaluation['Specificity'][key] = specificity(tn, fp)
        evaluation['False negative rate'][key] = false_negative_rate(fn, tp)
        evaluation['False positive rate'][key] = false_positive_rate(fp, tn)
        evaluation['F1'][key] = f1(tp, fp, fn)
        if key != 'DNN':
            evaluation['Cohen kappa'][key] = metrics.cohen_kappa_score(methods[key], y_true)
        else:
            evaluation['Cohen kappa'][key] = kappa.result().numpy()
    return evaluation



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
