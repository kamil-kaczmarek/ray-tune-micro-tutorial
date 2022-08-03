# Setup an environment
For this micro-tutorial we use conda environment manager and Python=3.8. Please follow the instructions to set it up.

## Step 1: Clone this repository
In your terminal run:
```
git clone https://github.com/kamil-kaczmarek/ray-tune-micro-tutorial.git
```

then, go to the repository directory. In your terminal run:
```
cd ray-tune-micro-tutorial
```

## Step 2: Create new environment and install dependencies

In your terminal run:
```
conda env create --file environment.yml
```
_(this can take a while)_

## Step 3: Activate an environment
```
conda activate ray_tune_demo
```

## Step 4: Install ray and tune
```
pip install "ray[default]" "ray[tune]"
```

🟩 Environment setup is done. You now have Ray and Tune installed. 🟩

----

# Test an environment
Here, you check if Ray is installed correctly. This is optional but recommended.

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

🟩 Congrats: ray is installed correctly. 🟩

# What to do next?
Please head over to the [Where to Start?](https://github.com/kamil-kaczmarek/ray-tune-micro-tutorial/blob/kk/dev/README.md#where-to-start) section in the README to start the tutorial!

----

# Troubleshooting
Please check [installation docs](https://docs.ray.io/en/latest/ray-overview/installation.html) for more details and [troubleshooting](https://docs.ray.io/en/latest/ray-overview/installation.html#troubleshooting) for issues with installing.
