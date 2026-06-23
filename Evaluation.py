import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import Utils
from sklearn.calibration import calibration_curve
from sklearn.model_selection import GroupShuffleSplit


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


