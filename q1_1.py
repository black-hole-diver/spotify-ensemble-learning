from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import arff

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    LabelEncoder, MinMaxScaler, OneHotEncoder
)
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.base import clone
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, f1_score
)

DATA = Path("data/spotify_26114941.arff")
RESULT = Path("results")
RESULT.mkdir(exist_ok=True)

data, meta_data = arff.loadarff(DATA)
df = pd.DataFrame(data)

for col in df.columns:
    if df[col].dtype is object:
        df[col] = df[col].map(
            lambda x: x.decode("utf-8")
            if isinstance(x, bytes) else x
        )

print("Data shape:", df.shape)
print("Class distribution:\n", df["playlist_genre"].value_counts())
print("Missing values:\n", df.isnull().sum())