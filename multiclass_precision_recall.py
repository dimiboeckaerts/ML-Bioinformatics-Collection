def multiclass_precision_recall(y_true, probas_pred, classes, make_plot=False):
    """
    A multiclass extension of the 'precision_recall_curve' function in Scikit-learn. Also includes a plotting
    functionality and outputs the PR AUC score.
    
    Dependencies: matplotlib.pyplot, sklearn.preprocessing, sklearn.metrics
    
    Input:
    - y-true: true labels
    - probas_pred: multiclass probabilities
    - classes: unique classes needed for binarization (not class weights or counts, see label_binarize in sklearn)
    - make_plot: boolean (default: False)
    
    Output: if plot=False (default): P, R and AUC for every class (dicts); else: a figure.
    """
    # define needed variables
    y_true_binary = label_binarize(y_true, classes)
    n_classes = len(classes)
    precision = dict()
    recall = dict()
    auc_scores = dict()
    
    # calculate P, R and AUC per class
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_true_binary[:, i], probas_pred[:, i])
        auc_scores[i] = auc(recall[i], precision[i])
    
    # make plot if asked and return results
    if make_plot:
        fig, ax = plt.subplots(figsize=(10,7))
        for i in range(n_classes):
            ax.plot(recall[i], precision[i], lw=2)
            
        ax.set_xlabel('Recall', size=12)
        ax.set_ylabel('Precision', size=12)
        ax.legend(classes)
        fig.tight_layout()
        return fig
    
    else:
        return precision, recall, auc_scores
