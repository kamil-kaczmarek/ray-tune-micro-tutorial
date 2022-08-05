# Micro tutorial on how to run and scale hyperparameter optimization with LightGBM and Tune

![tune_overview](https://docs.ray.io/en/latest/_images/tune_overview.png)

### It is for you if:
* you model structured data using LightGBM (classification or regression tasks).
* want to run or scale hyperparameter optimization in your project.
* Looking around for quick ways to try hyperparameter optimization in your project.

### What will you do?
* Run HPO job with LightGBM on the structured data.
* Configure _search algorithm_ and _scheduler_ for more efficient tuning: faster convergence and early stopping.
* Configure Tune to better utilize available compute resources.

### What will you learn?
* Few bits about Ray and Tune fundamentals.
* How to use Tune to run hyperparameter optimization - quick start.
* Few more bits about _search algorithm_ and _scheduler_ - to better define how tuning should progress.

# Where to start?
* First, make sure that you have an environment ready. Please follow the instructions on [environment setup](environment_setup.md) page (5 minutes read).
* Once your env is ready, go ahead and start [ray_tune_micro_tutorial.ipynb](ray_tune_micro_tutorial.ipynb).

# What to do next?
* Check the [user guides](https://docs.ray.io/en/latest/tune/tutorials/overview.html) for more in-depth introduction to Tune.
* Learn about [Distributed LightGBM on Ray](https://docs.ray.io/en/latest/ray-more-libs/lightgbm-ray.html) that enables multi-node and multi-GPU training.
* Have a closer look at [Tune docs](https://docs.ray.io/en/latest/tune/index.html) to learn more about other [search algorithms](https://docs.ray.io/en/latest/tune/api_docs/suggestion.html) and [schedulers](https://docs.ray.io/en/latest/tune/api_docs/schedulers.html).
* Go to the Ray tutorials page to 

# How to connect with community, learn more, join other trainings?
* Feel free to reach out on [Ray-distributed Slack](https://ray-distributed.slack.com/archives/C011ML23W5B). Join `#tutorials` channel, say hello and ask questions.
