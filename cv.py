from sklearn import model_selection

# :: source https://github.com/abhishekkrthakur/mlframework/
"""
- -- binary classification
- -- multi class classification
"""


class CrossValidation:
    def __init__(
            self,
            df, 
            target_cols,
            shuffle, 
            problem_type="binary_classification",
            multilabel_delimiter=",",
            num_folds=5,
            random_state=42
        ):
        self.dataframe = df
        self.target_cols = target_cols
        self.num_targets = len(target_cols)
        self.problem_type = problem_type
        self.num_folds = num_folds
        self.shuffle = shuffle,
        self.random_state = random_state
        self.multilabel_delimiter = multilabel_delimiter

        if self.shuffle is True:
            self.dataframe = self.dataframe.sample(frac=1).reset_index(drop=True)
        
        self.dataframe["kfold"] = -1
    
    def split(self):
        if self.problem_type in ("binary_classification", "multiclass_classification"):
            if self.num_targets != 1:
                raise Exception("Invalid number of targets for this problem type")
            target = self.target_cols[0]
            unique_values = self.dataframe[target].nunique()
            if unique_values == 1:
                raise Exception("Only one unique value found!")
            elif unique_values > 1:
                kf = model_selection.StratifiedKFold(n_splits=self.num_folds, 
                                                     shuffle=False)
                
                for fold, (train_idx, val_idx) in enumerate(kf.split(X=self.dataframe, y=self.dataframe[target].values)):
                    self.dataframe.loc[val_idx, 'kfold'] = fold

        return self.dataframe
