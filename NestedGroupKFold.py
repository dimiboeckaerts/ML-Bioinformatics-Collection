def NestedGroupKFold(model, X, y, parameter_grid, groups, class_weights, cv_score=make_scorer(accuracy_score),
                     eval_score=f1_score, inner_cv=GroupKFold(n_splits=4), outer_cv=GroupKFold(n_splits=4)):
    """
    Implements a nested version of GroupKFold cross-validation using GridSearchCV to evaluate models 
    that need hyperparameter tuning in settings where different groups exist in the available data.
    
    Dependencies: sklearn.model_selection, sklearn.metrics, numpy
    
    Input:
    - X, y: features and labels (must be NumPy arrays).
    - model, parameter_grid: the model instance and its parameter grid to be optimized.
    - groups: the groups to use in both inner- and outer loop.
    - class_weights: class weights to account for class imbalance in performance measurements.
    - cv_score: the scoring to use in inner loop (default: accuracy).
    - eval_score: the scoring to use in outer loop; currently only F1, precision and recall are allowed (default: F1).
    - inner_cv, outer_cv: the iterators for both CV-loops (default: GroupKFold(n_splits=4)).
    
    Output: array of scores for each CV-run (same output as cross_val_score of scikit-learn).
    """

    # define empty matrix to store performances (n CV runs and four performance metrics)
    n_splits_outer = outer_cv.get_n_splits()
    performances = np.zeros((n_splits_outer))
    
    # define outer loop
    loop = 0
    for train_outer, test_outer in outer_cv.split(X, y, groups):
        X_train, X_test = X[train_outer], X[test_outer]
        y_train, y_test = y[train_outer], y[test_outer]
        groups_train, groups_test = groups_array[train_outer], groups_array[test_outer]
        
        # define inner loop (in GridSearchCV)
        tuned_model = GridSearchCV(model, cv=inner_cv, param_grid=parameter_grid, scoring=cv_score)
        tuned_model.fit(X_train, y_train, groups=groups_train)
        
        # make predictions for test set (outer loop)
        y_pred = tuned_model.predict(X_test)
        
        # evaluate performance (factoring in class imbalance)
        score_list = list(eval_score(y_test, y_pred, average=None))
        final_score = sum([a*b for a,b in zip(score_list, class_weights)])/sum(class_weights)
        performances[loop] = final_score
        
        # next loop
        loop += 1
    
    return performances    
