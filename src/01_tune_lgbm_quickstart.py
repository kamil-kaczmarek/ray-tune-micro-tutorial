import lightgbm as lgb
import numpy as np
from ray import tune
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

# Search space
search_space = {
    "objective": "multiclass",
    "metric": "multi_logloss",
    "num_class": 3,
    "num_leaves": tune.choice([2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 40, 100]),
    "learning_rate": tune.loguniform(1e-4, 1e-1),
    "feature_fraction": tune.uniform(0.85, 0.999),
    "bagging_fraction": 0.8,
    "bagging_freq": tune.randint(1, 11),
    "max_depth": tune.randint(1, 11),
}


# Trainable
def train_lgbm(training_parameters):
    # Train LightGBM model and log results to stdout
    gbm = lgb.train(
        params=training_parameters,
        train_set=train_data,
        num_boost_round=10,
        valid_sets=[train_data, valid_data],
        valid_names=["train", "valid"],
        callbacks=[lgb.log_evaluation()],
    )

    # Print accuracy on validation data
    y_pred = np.argmax(gbm.predict(X_valid), axis=1)
    acc = accuracy_score(y_true=y_valid, y_pred=y_pred)
    print(f"accuracy is: {acc:.4f}")

    # Send the current training result back to Tune
    tune.report(valid_acc=acc)


# Run hyperparameter tuning
analysis = tune.run(train_lgbm, config=search_space)

# Display acc from the best trial
df = analysis.dataframe(metric="valid_acc")
print(df[["valid_acc", "trial_id", "pid"]])
