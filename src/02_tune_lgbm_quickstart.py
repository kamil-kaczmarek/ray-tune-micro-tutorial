import lightgbm as lgb
import numpy as np
import ray
from ray import tune
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Initialize Ray cluster
if ray.is_initialized:
    ray.shutdown()
cluster_info = ray.init()
print(cluster_info.address_info)

# Prepare dataset
X, y = load_digits(return_X_y=True)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=7707)

# Search space
search_space = {
    "objective": "multiclass",
    "metric": "multi_logloss",
    "num_class": 10,
    "num_leaves": tune.choice([2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 40, 100]),
    "learning_rate": tune.loguniform(1e-4, 1e-1),
    "feature_fraction": tune.uniform(0.5, 0.999),
    "bagging_fraction": 0.5,
    "bagging_freq": tune.randint(1, 50),
    "max_depth": tune.randint(1, 11),
    "verbose": -1,
}


# Trainable
def train_lgbm(training_params, checkpoint_dir=None):
    train_data = lgb.Dataset(data=X_train, label=y_train, free_raw_data=False)

    # Initialize booster
    gbm = lgb.Booster(params=training_params, train_set=train_data)

    # Train booster for 200 iterations
    for i in range(200):
        gbm.update(train_set=train_data)

        y_pred = np.argmax(gbm.predict(X_valid), axis=1)
        acc = accuracy_score(y_true=y_valid, y_pred=y_pred)

        # Send accuracy back to Tune
        tune.report(valid_acc=acc)


# Run hyperparameter tuning, single trial
analysis = tune.run(train_lgbm, config=search_space)

# Display info about this trial
df = analysis.dataframe(metric="valid_acc")
print(df[["valid_acc", "trial_id", "pid"]])

# Shutdown Ray cluster
ray.shutdown()
