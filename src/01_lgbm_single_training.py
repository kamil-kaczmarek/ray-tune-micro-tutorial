import lightgbm as lgb
import numpy as np
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Prepare dataset
X, y = load_digits(return_X_y=True)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=7707)

train_data = lgb.Dataset(data=X_train, label=y_train, free_raw_data=False)
valid_data = lgb.Dataset(data=X_valid, label=y_valid, free_raw_data=False, reference=train_data)

# Set training parameters for single training run
training_parameters = {
    "objective": "multiclass",
    "metric": "multi_logloss",
    "num_class": 10,
    "num_leaves": 5,
    "learning_rate": 0.001,
    "feature_fraction": 0.5,
    "bagging_fraction": 0.5,
    "bagging_freq": 100,
    "max_depth": 2,
    "verbose": -1,
}

# Initialize booster
gbm = lgb.Booster(params=training_parameters, train_set=train_data)

# Train booster for 200 iterations
for i in range(200):
    gbm = lgb.train(
        init_model=gbm,
        params=training_parameters,
        train_set=train_data,
        num_boost_round=1,
        valid_sets=[train_data, valid_data],
        valid_names=["train", "valid"],
        callbacks=[lgb.log_evaluation(period=100)],
        keep_training_booster=True,
    )

# Print accuracy on validation data
y_pred = np.argmax(gbm.predict(X_valid), axis=1)
acc = accuracy_score(y_true=y_valid, y_pred=y_pred)
print(f"Accuracy on valid set: {acc:.4f}")
