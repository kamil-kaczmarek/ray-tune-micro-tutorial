# ----------------------------------------------------------------------------------------- #
# Micro tutorial on how to run and scale hyperparameter optimization with LightGBM and Tune #
# Aug 2022                                                                                  #
# San Francisco, CA                                                                         #
# ----------------------------------------------------------------------------------------- #

# ---------------------------------------- #
# Part 1: single LightGBM training session #
# ---------------------------------------- #

# Preliminaries
# Imports
import lightgbm as lgb
import numpy as np
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Prepare dataset
X, y = load_digits(return_X_y=True)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=7707)

train_data = lgb.Dataset(data=X_train, label=y_train, free_raw_data=False)

# Set training parameters for single training run
training_parameters = {
    "objective": "multiclass",
    "metric": "multi_logloss",
    "num_class": 10,
    "num_leaves": 5,
    "learning_rate": 0.001,
    "feature_fraction": 0.5,
    "bagging_fraction": 0.5,
    "bagging_freq": 50,
    "max_depth": 2,
    "verbose": -1,
}

# Initialize booster
gbm = lgb.Booster(params=training_parameters, train_set=train_data)

# Train booster for 200 iterations
for i in range(200):
    gbm.update(train_set=train_data)

# Print accuracy on validation data
y_pred = np.argmax(gbm.predict(X_valid), axis=1)
acc = accuracy_score(y_true=y_valid, y_pred=y_pred)
print(f"Accuracy on valid set: {acc:.4f}, after {gbm.current_iteration()} iterations.")


# ----------------------- #
# Part 2: Tune quickstart #
# ----------------------- #

# Initialize Ray cluster
import ray

if ray.is_initialized:
    ray.shutdown()
cluster_info = ray.init()
print(cluster_info.address_info)

# Import tune
from ray import tune

# Define search space
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


# Define trainable
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

# ----------------------------------------- #
# Part 3: Execute 100 tuning runs with Tune #
# ----------------------------------------- #

# Run hyperparameter tuning
analysis = tune.run(
    train_lgbm,
    config=search_space,
    num_samples=100,
    metric="valid_acc",
    resources_per_trial={"cpu": 1},
    verbose=1,
)

# Display accuracy from the best trials
df = analysis.dataframe(metric="valid_acc")
print(df[["valid_acc", "trial_id", "pid"]].sort_values(by=["valid_acc"], ascending=False).head(n=5))

# ---------------------- #
# Part 4: ASHA with Tune #
# ---------------------- #

# Import ASHA from Tune schedulers
from ray.tune.schedulers import ASHAScheduler

# Create ASHA scheduler (Asynchronous Successive Halving Algorithm)
asha = ASHAScheduler(time_attr="training_iteration", mode="max", grace_period=50)

# Run hyperparameter tuning with ASHA scheduler
analysis = tune.run(
    train_lgbm,
    config=search_space,
    num_samples=100,
    metric="valid_acc",
    resources_per_trial={"cpu": 1},
    scheduler=asha,
    verbose=1,
)

# Display accuracy from the best trials
df = analysis.dataframe(metric="valid_acc")
print(df[["valid_acc", "trial_id", "pid"]].sort_values(by=["valid_acc"], ascending=False).head(n=5))

# Shutdown Ray cluster
ray.shutdown()
