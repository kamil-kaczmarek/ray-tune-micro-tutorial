# Preliminaries
import lightgbm as lgb
import numpy as np
import ray
from sklearn.datasets import load_wine
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Create Ray cluster
if ray.is_initialized:
    ray.shutdown()
cluster_info = ray.init()
print(cluster_info.address_info)

# Prepare dataset
X, y = load_wine(return_X_y=True)
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=7707, stratify=y
)

train_data = lgb.Dataset(data=X_train, label=y_train)
valid_data = lgb.Dataset(data=X_valid, label=y_valid, reference=train_data)

# ---------------------------------------- #
# Part 1: single LightGBM training session #
# ---------------------------------------- #

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


# ----------------------- #
# Part 2: Tune quickstart #
# ----------------------- #

# Import Tune
from ray import tune

# Define search space
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
    "verbose": -1,
}


# Define trainable
def train_lgbm(training_params, checkpoint_dir=None):
    train_data = lgb.Dataset(data=X_train, label=y_train)
    valid_data = lgb.Dataset(data=X_valid, label=y_valid, reference=train_data)

    # Train LightGBM model and log results to stdout
    gbm = lgb.train(
        params=training_params,
        train_set=train_data,
        num_boost_round=10,
        valid_sets=[train_data, valid_data],
        valid_names=["train", "valid"],
        callbacks=[lgb.log_evaluation(period=10)],
    )

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
# Part 3: Execute 300 tuning runs with Tune #
# ----------------------------------------- #

# Run hyperparameter tuning
analysis = tune.run(
    train_lgbm,
    config=search_space,
    num_samples=300,
    metric="valid_acc",
    resources_per_trial={"cpu": 1},
)

# Display accuracy from the best trials
df = analysis.dataframe(metric="valid_acc")
print(df[["valid_acc", "trial_id", "pid"]].sort_values(by=["valid_acc"], ascending=False).head(n=5))

# ------------------------------------------- #
# Part 4: Population Based Training with Tune #
# ------------------------------------------- #

# Import Population Based Training from Tune schedulers
from ray.tune.schedulers import PopulationBasedTraining

# Create Population Based Training scheduler
pbt_scheduler = PopulationBasedTraining(
    time_attr="time_total_s",
    mode="max",
    perturbation_interval=3,
    hyperparam_mutations={
        "num_leaves": tune.choice([2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 40, 100]),
        "learning_rate": tune.loguniform(1e-4, 1e-1),
        "feature_fraction": tune.uniform(0.85, 0.999),
        "bagging_freq": tune.randint(1, 11),
        "max_depth": tune.randint(1, 11),
    },
)

# Run hyperparameter tuning
analysis = tune.run(
    train_lgbm,
    config=search_space,
    num_samples=300,
    metric="valid_acc",
    resources_per_trial={"cpu": 1},
    scheduler=pbt_scheduler,
)

# Display accuracy from the best trials
df = analysis.dataframe(metric="valid_acc")
print(df[["valid_acc", "trial_id", "pid"]].sort_values(by=["valid_acc"], ascending=False).head(n=5))

# Shutdown Ray cluster
ray.shutdown()
