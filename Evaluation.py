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


def pr(y_pred, y_true,titleappendix = ""):

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
    plt.title('Precision-Recall Curve - ' + titleappendix)
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


def plotleadingmatch(df, metrics, **kwargs):
    for metric in metrics:
            Utils.draw_feature(feature=metric, match_groups=Utils.build_match_groups(df.loc[df.groupby("mchID")[metric].idxmax()].reset_index(drop=True)), **kwargs)

def featuredecompositionplot(
    df,
    featureplot,
    featurebreakdown,
    equalwidth,
    n_bins=5,
    nbins=30,
    density=True,
    colours=None,
    title=None,
    log=True,
    ax=None,
    hist_kwargs=None,
):
    """Plots `featureplot` distribution broken down by bins of `featurebreakdown`.

    Uses Matplotlib `ax.hist` per bin so legend entries show counts and labels correctly.
    Args:
        df: DataFrame
        featureplot: column to histogram
        featurebreakdown: column to bin for breakdown
        equalwidth: if True use quantile-based binning (qcut), else equal-width (`cut`)
        n_bins: number of breakdown bins
        nbins: histogram bins for `ax.hist`
        density: pass to `ax.hist`
        colours: dict mapping bin -> color or None to use palette
        title: plot title
        log_x: if True set x-axis to log scale
        ax: Matplotlib axis (created if None)
        hist_kwargs: additional kwargs forwarded to `ax.hist`
    """

    df = df.copy()

    bins_col = f"{featurebreakdown}_bin"
    if equalwidth:
        df[bins_col] = pd.qcut(df[featurebreakdown], q=n_bins, duplicates="drop")
    else:
        df[bins_col] = pd.cut(df[featurebreakdown], bins=n_bins)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    # Determine categories in a stable order
    if pd.api.types.is_categorical_dtype(df[bins_col]):
        categories = list(df[bins_col].cat.categories)
    else:
        categories = sorted(df[bins_col].dropna().unique())

    palette = sns.color_palette(n_colors=max(3, len(categories)))
    hist_kwargs = hist_kwargs or {}

    # range across full data for consistent binning
    data_series = df[featureplot].dropna()
    if data_series.empty:
        raise ValueError(f"No data for feature '{featureplot}' to plot")
    data_min, data_max = data_series.min(), data_series.max()

    for i, cat in enumerate(categories):
        group = df[df[bins_col] == cat][featureplot].dropna()
        if len(group) == 0:
            continue
        color = None
        if isinstance(colours, dict):
            color = colours.get(cat, None)
        if color is None:
            color = palette[i % len(palette)]

        ax.hist(
            group,
            bins=nbins,
            range=(data_min, data_max),
            histtype="step",
            linewidth=2,
            alpha=0.8,
            density=density,
            color=color,
            label=f"{cat}  (n={len(group):,})",
            **hist_kwargs,
        )

    ax.set_ylabel("Density" if density else "Counts", fontsize=12)
    ax.set_xlabel(featureplot, fontsize=12)
    ax.set_title(title or f"{featureplot} by {featurebreakdown}", fontsize=14)
    ax.tick_params(axis="both", labelsize=10)
    ax.legend(fontsize=10, loc="best", frameon=False, title=f"{featurebreakdown} bins")
    if log:
        ax.set_yscale("log")
    ax.grid(True, which="both", ls="--", lw=0.5)
    plt.tight_layout()
    plt.show()



