import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import Utils
from sklearn.calibration import calibration_curve
from sklearn.model_selection import GroupShuffleSplit
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, auc, precision_recall_curve
import onnxruntime as ort



def PeekData (df : pd.DataFrame) -> None:
    """High Level evaluation of data we're working with"""
    n_mch_tracks = df["mchID"].nunique()
    n_positive = df["IsSignal"].sum()
    candidates_per_track = df.groupby("mchID").size()

    print(f"MCH tracks:          {n_mch_tracks:,}")
    print(f"Total pairs:         {len(df):,}")
    print(f"True matches:        {int(n_positive):,} ({100*n_positive/len(df):.1f}%)")
    print(f"Candidates per track: min={candidates_per_track.min()}, "
        f"max={candidates_per_track.max()}, "
        f"mean={candidates_per_track.mean():.2f}")

    # Tracks with no true match among candidates
    tracks_with_match = df.groupby("mchID")["IsSignal"].max()
    n_no_match = (tracks_with_match == 0).sum()
    print(f"Tracks with no true match in candidates: {n_no_match:,} "
        f"({100*n_no_match/n_mch_tracks:.1f}%)")
    print('Data loaded and preprocessed. Ready for training.')
    return

def MakeCategories(df :pd.DataFrame) -> pd.DataFrame:
    """Constructs the 4 categories for classification approaches"""
    CATEGORY_NAMES = ["True", "Wrong", "Decay", "Fake"]
    raw_to_category = {}
    for category_name in CATEGORY_NAMES:
        for raw_value in Utils.MATCH_LABEL_GROUPS[category_name]:
            raw_to_category[raw_value] = CATEGORY_NAMES.index(category_name)

    df['MatchLabel_Category'] = df['MatchLabel'].map(raw_to_category).fillna(-1).astype(int)

    print('Label mapping order:', CATEGORY_NAMES)
    print('Label distribution:')
    print(df['MatchLabel_Category'].value_counts().sort_index())
    print('\nLabel mapping: 0=True, 1=Wrong, 2=Decay, 3=Fake')
    return df

def Splitter(df : pd.DataFrame, val_frac : float, test_frac : float):
    """Provides the splitting of DFs into group preserved df's for Train, Validation, and Test Respectively.
    Performs this for both categorical and binary approaches"""


    valsubfrac = val_frac/(1-test_frac)       
    groups = df["mchID"].values

    test_splitter = GroupShuffleSplit(n_splits=1, test_size=test_frac, random_state=42)
    temp_idx, test_idx = next(test_splitter.split(df, groups=groups))

    df_temp = df.iloc[temp_idx]
    df_test = df.iloc[test_idx]

    temp_groups = df_temp["mchID"].values
    val_splitter = GroupShuffleSplit(n_splits=1, test_size=valsubfrac, random_state=42)
    train_idx, val_idx = next(val_splitter.split(df_temp, groups=temp_groups))

    df_train = df_temp.iloc[train_idx]
    df_val = df_temp.iloc[val_idx]

    return df_train, df_val, df_test


def splitter_internal( df:pd.DataFrame, test_frac:float):
    """internally used splitter, also useful for when we have separate train and test datasets
    i.e. weighted input ones"""
          
    groups = df["mchID"].values

    test_splitter = GroupShuffleSplit(n_splits=1, test_size=test_frac, random_state=42)
    temp_idx, test_idx = next(test_splitter.split(df, groups=groups))

    df_temp = df.iloc[temp_idx]
    df_test = df.iloc[test_idx]

    return df_temp, df_test



def cm(y_pred, y_true, normalize=None):
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred, normalize=normalize)
    label_names = ["Not True","True"]

    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt='f', cmap='Blues', 
                xticklabels=label_names, yticklabels=label_names,
                ax=ax, cbar_kws={'label': 'Proportion' if normalize else 'Count'})
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('Confusion Matrix - Test Set')
    plt.tight_layout()
    plt.show()

    # Print classification metrics
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=label_names))
    print(f"\nOverall Accuracy: {accuracy_score(y_true, y_pred):.4f}")

    # Per-class metrics
    print("\nPer-class Performance:")
    for i, label in enumerate(label_names):
        tn = cm.sum() - cm[i].sum() - cm[:, i].sum() + cm[i, i]
        tp = cm[i, i]
        fp = cm[:, i].sum() - cm[i, i]
        fn = cm[i].sum() - cm[i, i]
        
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"  {label:<8} - Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")


def pr(y_pred, y_true):

    precision, recall, _ = precision_recall_curve(y_true=y_true, y_score=y_pred)
    auc_pr = auc(recall, precision)
    positive_rate = y_true.mean()

    # Ideal starts at (0,1), moves to (1,1) for perfect recall, then drops to (1, proportion of positives)
    ideal_recall = [0, 1, 1]
    ideal_precision = [1, 1, positive_rate]

    # Random model baseline is a horizontal line at the positive class fraction
    random_recall = [0, 1]
    random_precision = [positive_rate, positive_rate]

    plt.figure(figsize=(7, 5))
    plt.plot(recall, precision, label='Actual Model', color='blue', lw=2)
    plt.plot(random_recall, random_precision, label='Random Baseline', color='gray', linestyle=':', lw=2)
    plt.plot(ideal_recall, ideal_precision, label='Ideal Model', color='green', linestyle='--', lw=2)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left')
    plt.grid(True)
    plt.text(0.95, 0.05, f'AUC-PR = {auc_pr:.4f}', ha='right', va='bottom', transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
    plt.show()
    print(f"Area under Precision-Recall Curve (AUC-PR): {auc_pr:.4f}")

def onnxinferxgb(df, features, model, targetname):
    sess = ort.InferenceSession(model)
    input_name = sess.get_inputs()[0].name
    df[targetname] = sess.run(
        None,
        {input_name: df[features].to_numpy(dtype=np.float32)}
    )[0]
    return df

def onnxinferlgbm(df, features, model, targetname):
    sess = ort.InferenceSession(model)
    input_name = sess.get_inputs()[0].name
    pred = sess.run(
        None,
        {input_name: df[features].to_numpy(dtype=np.float32)}
    )
    df[targetname] = [p[1] for p in pred[1]]
    return df