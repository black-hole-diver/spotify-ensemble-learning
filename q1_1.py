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

# Features and target class

encoder = LabelEncoder()
X = df.drop(columns=["playlist_genre"])
y = encoder.fit_transform(df["playlist_genre"])
class_names = encoder.classes_

numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = [ col for col in X.columns
                        if col not in numeric_features]

for col in ["key", "mode"]:
    if col in numeric_features:
        numeric_features.remove(col)
        categorical_features.append(col)

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", MinMaxScaler())
])
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])
preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])

models = {
    "NN": MLPClassifier(hidden_layer_sizes=(32,), max_iter=500, random_state=42),
    "1-kNN": KNeighborsClassifier(n_neighbors=1),
    "Decision Tree": DecisionTreeClassifier(random_state=42)
}
pipelines = {
    name: Pipeline([
        ("preprocessing", clone(preprocessor)),
        ("classifier", model)
    ]) for name, model in models.items()
}

CV = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

def evaluate_model(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "F1 Score": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "Macro F1 Score": f1_score(y_true, y_pred, average="macro", zero_division=0)
    }

predictions = {
    name: np.empty(len(y), dtype=int) for name in models
}

probabilities = {
    name: np.zeros((len(y), len(class_names))) for name in models
}



