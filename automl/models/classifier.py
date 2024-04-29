from time import time
import datetime
import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
from optuna.samplers import TPESampler
import shap
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
from sklearn.metrics import roc_auc_score

from automl.util import logger

class CrossLgbBinaryClassifier:
    def __init__(self, params=None, n_fold=5):
        self.models = []
        self.feature_importances_ = pd.DataFrame()
        self.n_fold = n_fold
        self.params_ = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'learning_rate': 0.01,
            'num_leaves': 2 ** 5,
            'bagging_fraction': 0.95,
            'bagging_freq': 1,
            'bagging_seed': 66,
            'feature_fraction': 0.7,
            'feature_fraction_seed': 66,
            'max_bin': 100,
            'max_depth': 5,
            'verbose': -1
        }
        if params is not None:
            self.params_ = params
        self.early_stop_round = 150
        self.num_boost_round = 8000
        self.verbose = 100

    def optuna_tuning(self, X, y, Debug=False):
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, stratify=y, test_size=0.2, random_state=202)

        def objective(trial):
            param_grid = {
                'num_leaves': trial.suggest_int('num_leaves', 2 ** 3, 2 ** 9),
                'num_boost_round': trial.suggest_int('num_boost_round', 100, 8000),
                'max_depth': trial.suggest_int('max_depth', 3, 9),
                'objective': 'binary',
                'metric': 'auc',
                'boosting_type': 'gbdt',
                'learning_rate': 0.01,
                'bagging_fraction': 0.95,
                'bagging_freq': 1,
                'bagging_seed': 66,
                'feature_fraction': 0.7,
                'feature_fraction_seed': 66,
                'max_bin': 100,
                'verbose': -1
            }
            train_data = lgb.Dataset(X_train, label=y_train, categorical_feature="")
            valid_data = lgb.Dataset(X_valid, label=y_valid, categorical_feature="")
            clf = lgb.train(param_grid, train_data, valid_sets=[train_data, valid_data],
                            callbacks=[lgb.early_stopping(self.early_stop_round)])
            pred_val = clf.predict(X_valid)
            auc_ = roc_auc_score(y_valid, pred_val)

            return auc_

        train_time = 1 * 10 * 60  # h * m * s
        if Debug:
            train_time = 1 * 1 * 60  # h * m * s
        study = optuna.create_study(direction='maximize', sampler=TPESampler(), study_name='LgbClassifier')
        study.optimize(objective, timeout=train_time)

        logger.info(f'Number of finished trials: {len(study.trials)}')
        logger.info('Best trial:')
        trial = study.best_trial

        logger.info(f'\tValue: {trial.value}')
        logger.info('\tParams: ')
        for key, value in trial.params.items():
            logger.info('\t\t{}: {}'.format(key, value))

        self.params_['num_leaves'] = trial.params['num_leaves']
        self.params_['max_depth'] = trial.params['max_depth']
        self.num_boost_round = trial.params['num_boost_round']
    
    def fit(self, X, y, early_stop_round=None, num_boost_round=None, verbose=None, tuning=True, 
            feature_importance_type='shap', Debug=False):
        logger.info(X.shape)

        if tuning:
            logger.info("[+]tuning params")
            self.optuna_tuning(X, y, Debug=Debug)

        if early_stop_round is not None:
            self.early_stop_round = early_stop_round
        if num_boost_round is not None:
            self.num_boost_round = num_boost_round
        if verbose is not None:
            self.verbose = verbose

        folds = StratifiedKFold(n_splits=self.n_fold, shuffle=True, random_state=2024)
        auc = []
        self.feature_importances_['feature'] = X.columns

        for fold, (train_idx, valid_idx) in enumerate(folds.split(X, y)):

            start_time = time()
            print(f'Training on fold {fold + 1}')

            train_data = lgb.Dataset(X.iloc[train_idx], label=y.iloc[train_idx])
            valid_data = lgb.Dataset(X.iloc[valid_idx], label=y.iloc[valid_idx])
            model = lgb.train(self.params_, train_data, num_boost_round=self.num_boost_round, valid_sets=[train_data,valid_data],
                              callbacks=[lgb.log_evaluation(self.verbose), lgb.early_stopping(self.early_stop_round)])
            self.models.append(model)
            # 特征重要性
            if feature_importance_type == 'shap':
                explainer = shap.TreeExplainer(model)
                shap_vals = explainer.shap_values(X.iloc[valid_idx])
                shap_data = np.sum(np.mean(abs(np.array(shap_vals)), axis=1), axis=0)
                self.feature_importances_[f'fold_{fold + 1}_shap_val'] = shap_data
            elif feature_importance_type in ['split', 'gain']:
                self.feature_importances_[f'fold_{fold + 1}_{feature_importance_type}'] = model.feature_importance(feature_importance_type)
            else:
                raise ValueError(f'unexpected feature importace type {feature_importance_type}')

            val = model.predict(X.iloc[valid_idx])
            auc_ = roc_auc_score(y.iloc[valid_idx], val)
            print(f'AUC: {auc_}')
            auc.append(auc_)
            print(f'Fold {fold + 1} finished in {str(datetime.timedelta(seconds=time() - start_time))}')
        self.feature_importances_['average'] = self.feature_importances_[
            [x for x in self.feature_importances_.columns if x != "feature"]].mean(axis=1)
        self.feature_importances_ = self.feature_importances_.sort_values(by="average", ascending=False)
        self.feature_importances_.index = range(len(self.feature_importances_))