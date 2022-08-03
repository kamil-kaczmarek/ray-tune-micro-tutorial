import lightgbm as lgb
import numpy as np
from sklearn.datasets import load_wine
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Prepare dataset
X, y = load_wine(return_X_y=True)
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=7707, stratify=y
)

train_data = lgb.Dataset(data=X_train, label=y_train)
valid_data = lgb.Dataset(data=X_valid, label=y_valid, reference=train_data)

# Set training parameters for single training run
training_parameters = {
    "objective": "multiclass",
    "metric": "multi_logloss",
    "num_class": 3,
    "num_leaves": 7,
    "learning_rate": 0.1,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "max_depth": 5,
    "verbose": -1,
}

# Train LightGBM model and log results to stdout
gbm = lgb.train(
    params=training_parameters,
    train_set=train_data,
    num_boost_round=10,
    valid_sets=[train_data, valid_data],
    valid_names=["train", "valid"],
    callbacks=[lgb.log_evaluation(period=10)],
)

# Print accuracy on validation data
y_pred = np.argmax(gbm.predict(X_valid), axis=1)
acc = accuracy_score(y_true=y_valid, y_pred=y_pred)
print(f"Accuracy on valid set: {acc:.4f}")
