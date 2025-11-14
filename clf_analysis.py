import itertools
import os
import warnings
from typing import Dict, Sequence

import numpy as np
import pandas as pd
from absl import app, flags
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from tqdm import tqdm

warnings.filterwarnings("ignore")

# The following clf and shattering code are adapted from PoissonVAE repository:
# https://github.com/hadivafaii/PoissonVAE

def clf_analysis_simple(
    mode: str,
    z: Dict[str, np.ndarray],
    y: Dict[str, np.ndarray],
    clf_type: str = 'knn',
    verbose: bool = False,
    **kwargs,
):
    def _get_clf(**kws):
        if clf_type == 'logreg':
            return LogisticRegression(**kws)
        elif clf_type == 'svm':
            return LinearSVC(dual='auto', **kws)
        elif clf_type == 'knn':
            return KNeighborsClassifier(**kws)
        else:
            raise NotImplementedError(clf_type)

    df = []
    if mode == 'clf':
        clf = _get_clf(**kwargs)
        clf.fit(z['trn'], y['trn'])
        pred = clf.predict(z['vld'])
        report = classification_report(
            y_true=y['vld'],
            y_pred=pred,
            output_dict=True,
        )
        df.append({
            'classifier': [type(clf).__name__],
            'accuracy': [report['accuracy']],
        })

    elif mode == 'shatter':
        digits = y['trn'].astype(int)
        all_labels = sorted(np.unique(digits))
        groups = disjoint_groups(all_labels)

        for idx, (c0, c1) in tqdm(
            enumerate(groups),
            disable=not verbose,
            total=len(groups),
            desc=clf_type,
            ncols=80
        ):
            y_trn_bin = digit2category(y['trn'], c1)
            y_vld_bin = digit2category(y['vld'], c1)

            clf = _get_clf(**kwargs)
            clf.fit(z['trn'], y_trn_bin)
            pred = clf.predict(z['vld'])

            report = classification_report(
                y_true=y_vld_bin,
                y_pred=pred,
                output_dict=True,
            )
            df.append({
                'group_idx': [idx],
                'category_0': [c0],
                'category_1': [c1],
                'classifier': [type(clf).__name__],
                'accuracy': [report['accuracy']],
            })
    else:
        raise NotImplementedError(mode)

    return pd.DataFrame(merge_dicts(df))

def disjoint_groups(data: Sequence):
    assert len(data) % 2 == 0, "# elements must be even"
    all_combos = list(itertools.combinations(data, len(data) // 2))
    seen = set()
    result = []

    for combo in all_combos:
        complement = tuple(sorted(set(data) - set(combo)))
        pair = (tuple(sorted(combo)), complement)
        if pair not in seen and (complement, tuple(sorted(combo))) not in seen:
            seen.add(pair)
            result.append((list(combo), list(complement)))

    return result

def digit2category(digits: np.ndarray, category: list):
    return np.isin(digits.astype(int), category).astype(int)

def merge_dicts(dicts: list) -> dict:
    from collections import defaultdict
    merged = defaultdict(list)
    for d in dicts:
        for k, v in d.items():
            merged[k].extend(v)
    return merged

def evaluate_on_subsets(
    z: np.ndarray,
    y: np.ndarray,
    train_sizes=[200, 1000, 5000],
    test_size=5000,
    mode='clf',
    clf_type='logreg',
    random_seed=42,
    verbose=False,
    **kwargs
):
    np.random.seed(random_seed)
    total = len(z)
    assert total >= max(train_sizes) + test_size, "Not enough data for train+test"

    indices = np.random.permutation(total)
    train_pool = indices[:max(train_sizes)]
    test_indices = indices[max(train_sizes):max(train_sizes)+test_size]

    z_test = z[test_indices]
    y_test = y[test_indices]

    results = []

    for train_size in train_sizes:
        train_indices = train_pool[:train_size]
        z_train = z[train_indices]
        y_train = y[train_indices]

        z_dict = {'trn': z_train, 'vld': z_test}
        y_dict = {'trn': y_train, 'vld': y_test}

        df = clf_analysis_simple(
            mode=mode,
            z=z_dict,
            y=y_dict,
            clf_type=clf_type,
            verbose=verbose,
            **kwargs
        )
        df['train_size'] = train_size
        results.append(df)

    return pd.concat(results, ignore_index=True)


def train_clf_analysis(z,
                       y,
                       clf_type="knn"):
    print("satrt classifying...")
    for train_size in [200,1000,5000]:
        acc_list = []
        for i in range(5):

            result_df = evaluate_on_subsets(
                z=z,
                y=y, 
                train_sizes=[train_size],
                test_size=5000,
                random_seed = i*100,
                mode='clf',  
                clf_type=clf_type
            )
            acc_list.append(result_df['accuracy'].mean())
        
        print("Train size {}: {:.3f}+/-{:.3f}".format(train_size, np.mean(acc_list), np.std(acc_list)))

    print("start shattering...")

    for train_size in [200]:
        acc_list = []
        for i in range(5):

            result_df = evaluate_on_subsets(
                z=z,  
                y=y,  
                train_sizes=[train_size],
                test_size=5000,
                random_seed = i*10,
                mode='shatter',  
                clf_type=clf_type
            )
            acc_list.append(result_df['accuracy'].mean())

        print("Train size {}: {:.3f}+/-{:.3f}".format(train_size, np.mean(acc_list), np.std(acc_list)))




