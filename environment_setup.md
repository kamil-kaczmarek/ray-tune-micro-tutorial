# Setup an environment
For this micro-tutorial we use conda environment manager and Python=3.8. Please follow the instructions to set it up.

## Step 1: Create new environment and install dependencies

In your terminal run:
```
conda create --name ray_tune_demo python=3.8.13 optuna grpcio scikit-learn lightgbm numpy
```

## Step 2: Activate an environement
```
conda activate ray_tune_demo
```

## Step 3: Install ray and tune
```
pip install "ray[default]" "ray[tune]"              
```

🟩 Environment setup is done. You now have Ray and Tune installed. 🟩

----

# (optional, recommended) Test an environment
Here, you check if Ray is installed correctly.

## Step 1: Start Python in the interactive mode
In your terminal run:
```
python3
```
Wait until you see `>>>` prompt. Now Python is ready :snake:

## Step 2: Import ray
```
>>> import ray
```

## Step 3: Start Ray runtime on your laptop
```
>>> ray.init()
```

You should see output like this:
```
2022-08-02 15:25:26,966 INFO services.py:1470 -- View the Ray dashboard at http://127.0.0.1:8265
```
Feel free to open Ray dashboard at [http://127.0.0.1:8265](http://127.0.0.1:8265)

🟩 Congrats: ray is installed correctly. Plese move on the the [README](README.md) to follow next steps in the tutorial! 🟩

----

# Troubleshooting
Please check [installation docs](https://docs.ray.io/en/latest/ray-overview/installation.html) for more details and [troubleshooting](https://docs.ray.io/en/latest/ray-overview/installation.html#troubleshooting) for issues with installing.
