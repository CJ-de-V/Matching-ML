import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import Utils
from sklearn.calibration import calibration_curve


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

def Splitter(df : pd.DataFrame, categorical : bool, train : float, val: float, test : float):
    """Provides the splitting of DFs into group preserved df's for Train, Validation, and Test Respectively.
    Performs this for both categorical and binary approaches"""
    leftover = 1-train
    valsubfrac = val/leftover   
    testsubfrac = test/leftover
    if( categorical):
        #usual splitting here
        return
    #binary approach stuffs here
    return 'home'