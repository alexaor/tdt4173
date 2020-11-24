import evaluation.utils as utils
import matplotlib.pyplot as plt
import numpy as np

from sklearn import metrics

from tensorflow_addons.metrics import CohenKappa


def print_evaluation(y_true, methods, filename="", dnn_conf_matrix=None) -> None:
    """
    Iterate through all methods in 'methods' and for each finds the confusion matrix and calculate the
    accuracy, misclassification, precision, recall, specificity, false positive rate, false negative rate and f1.
    When done, the evaluation will be printed out to terminal.

    Parameters
    ----------
    y_true : array
        An array with the true prediction to match with predictions from the methods
    methods : dict
        A dictionary, [method name, 1d binary array], the predicted output values for a method
    filename : string
        Name of file to save the evaluation, if empty nothing will be saved
    dnn_conf_matrix : list of ints
        The confusion matrix to the DNN, where the list is sorted by: tn, fp, fn, tp
    """

    output = ""
    methods = methods.copy()
    if dnn_conf_matrix is not None:
        kappa = CohenKappa(num_classes=2)
        kappa.update_state(y_true, methods['DNN'])
        methods['DNN'] = dnn_conf_matrix
    for key in methods.keys():
        if key == 'DNN':
            tn, fp, fn, tp = dnn_conf_matrix
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



def plot_roc_auc(y_true, methods, filename) -> None:
    """
    Creates a ROC curve with the methods,and calculate the AUC for each methods. The plot will be saved in the
    directory: 'results/plots'.

    Parameters
    ----------
    y_true : array
        An array with the true prediction to match with predictions from the methods
    methods : dict
        A dictionary, [method name, 1d binary array], the predicted output values for a method
    filename : string
        Name of file to save the evaluation
    """

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
    """
    Creates a Precision Recall curve with the methods, and calculating the absolute precision for each methods.
    The plot will be saved in the directory: 'results/plots'.

    Parameters
    ----------
    y_true : array
        An array with the true prediction to match with predictions from the methods
    methods : dict
        A dictionary, [method name, 1d binary array], the predicted output values for a method
    filename : string
        Name of file to save the evaluation
    """

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
    ax.set_xlim(0.15, 1.05)
    # Save plot
    file_path = utils.save_plot(fig, filename)
    print(f"Saved Precision Recall curve at: '{file_path}'")


def plot_evaluation_result(y_true, methods, dirname="", dnn_conf_matrix=None) -> None:
    """
    Makes a separate plot for each evaluation method, where all the methods in the dictionary 'methods' are being
    compared to one another. The plots will be saved in the directory: 'results/plots/<dirname>'.

    Parameters
    ----------
    y_true : array
        An array with the true prediction to match with predictions from the methods
    methods : dict
        A dictionary, [method name, 1d binary array], the predicted output values for a method
    dirname : string
        Name of directory where the plots should be saved
    dnn_conf_matrix : list of ints
        The confusion matrix to the DNN, where the list is sorted by: tn, fp, fn, tp
    """
    methods = methods.copy()
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


def plot_comparison(y_true, methods, evallist, filename, dnn_conf_matrix=None) -> None:
    """
    Compares the methods in the methods dictionary with the wanted evaluations method. The plot will be saved in the
    directory: 'results/plots'.

    Parameters
    ----------
    y_true : array
        An array with the true prediction to match with predictions from the methods
    methods : dict
        A dictionary, [method name, 1d binary array], the predicted output values for a method
    evallist : list of strings
        A list of the evaluations the methods will be compared against
    filename : string
        Name of directory where the plots should be saved
    dnn_conf_matrix : list of ints
        The confusion matrix to the DNN, where the list is sorted by: tn, fp, fn, tp
    """

    methods = methods.copy()
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
    ax.legend(loc='best', prop={'size': 20})
    fig.tight_layout()
    ax.set_ylim(0, 90)
    # Save or show plots
    plot_dir = utils.save_plot(fig, filename)
    print(f"Saved the comparison plot in directory: '{plot_dir}/{filename}'")


def get_all_evaluations(y_true, methods, dnn_conf_matrix) -> dict:
    """
    Creates a new dictionary where the evaluations method is the key, the value for the keys are a new dictionary, where
    the keys are the method and the values are the result for the given evaluation method. This dictionary is returned.

    Parameters
    ----------
    y_true : array
        An array with the true prediction to match with predictions from the methods
    methods : dict
        A dictionary, [method name, 1d binary array], the predicted output values for a method
    dnn_conf_matrix : list of ints
        The confusion matrix to the DNN, where the list is sorted by: tn, fp, fn, tp

    Returns
    -------
    evaluation : dict
        A dictionary of type [str, dict[str, float]], which is the results for the different evaluations methods
    """
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
                  'Cohen kappa': {},
                  'Number of yes': {}}
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
        evaluation['Number of yes'][key] = (fp + tp)/(tn + fp + fn + tp)
    return evaluation


def accuracy(tp, tn, total) -> float:
    """
    Calculating and returning the accuracy for the given case.

    Parameters
    ----------
    tp : int
        Number of true positives
    tn : int
        Number of true negatives
    total : int
        Total number of cases

    Returns
    -------
    accuracy : float
        The accuracy of the prediction
    """
    return (tp + tn)/total


def misclassification(tp, tn, total) -> float:
    """
    Calculating and returning the miss classification, the percentage of cases that was wrongly classified.

    Parameters
    ----------
    tp : int
        Number of true positives
    tn : int
        Number of true negatives
    total : int
        Total number of cases

    Returns
    -------
    misclassification : float
        The miss classifications of the prediction
    """
    return 1 - accuracy(tp, tn, total)


def precision(tp, fp):
    """
    Calculating and returning the precision, out of total predicted true,the percentage of how often the model predict
    accurate.

    Parameters
    ----------
    tp : int
        Number of true positives
    fp : int
        Number of false positives

    Returns
    -------
    precision : float
        The precision of the prediction
    """
    return tp/(tp + fp)


def recall(tp, fn):
    """
    Calculating and returning the recall (true positive rate), number of items correctly identified as positive out of
    total true positives

    Parameters
    ----------
    tp : int
        Number of true positives
    fn : int
        Number of false negatives

    Returns
    -------
    recall : float
        The recall of the prediction
    """
    return tp/(tp + fn)


def specificity(tn, fp):
    """
    Calculating and returning the specificity (true negative rate), number of items correctly identified as negative out
    of total negatives.

    Parameters
    ----------
    tn : int
        Number of true negatives
    fp : int
        Number of false positives

    Returns
    -------
    specificity : float
        The specificity of the prediction
    """
    return tn/(tn + fp)


def false_positive_rate(fp, tn):
    """
    Calculating and returning the fpr, the fraction of all negative instances that the classifier incorrectly
    identifies as positive.

    Parameters
    ----------
    fp : int
        Number of false positives
    tn : int
        Number of true negatives

    Returns
    -------
    false_positive_rate : float
        The false positive rate of the prediction
    """
    return fp/(fp + tn)


def false_negative_rate(fn, tp):
    """
    Calculating and returning the fnr, number of items wrongly identified as negative out of total true positives.

    Parameters
    ----------
    fn : int
        Number of false negatives
    tp : int
        Number of true positives

    Returns
    -------
    false_negative_rate : float
        The false negative rate of the prediction
    """
    return fn/(fn + tp)


def f1(tp, fp, fn):
    """
    Calculating and returning the f-measure, a score that combines precision recall into a single number.

    Parameters
    ----------
    tp : int
        Number of true positives
    fp : int
        Number of false positives
    fn : int
        Number of false negatives

    Returns
    -------
    f1 : float
        The f-measure of the prediction
    """
    p = precision(tp, fp)
    r = recall(tp, fn)
    return (2 * r * p)/(r + p)
