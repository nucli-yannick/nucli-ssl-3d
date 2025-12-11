

# Nucli-Train
This repository contains Nuclivision's main framework & code for training deep-learning models. Our goal is to offer a modular framework, such that users only need to bother with aspects relevant to their projects and goals.


## Installing
To install, run 
`
    pip install -e .
` and optionally install dash & plotly to use interactive visualization tools.

## Training
The `Trainer` object handles things like mixed precision, multi-device training, automated logging to mlflow, etc. It has a number of arguments, specifying:
- model object, which contains the loss functions, network(s), and any modeling specific code (e.g when training a GAN)
- A training dataloader & validation loaders
- experiment & run name, logging intervals, ..
- options specifying computational aspects (e.g whether model gets compiled, AMP, multi-gpu training, ...)
- Optionally a resuming argument, which allows you to resume training from a checkpoint or initiate model weights from a previous run

A detailed description of each argument and how they function internally is given in `/docs`, but the general flow is displayed through a basic example:

```python
from nucli_train import Trainer


model = build_model(cfg) # builds the networks, losses, validation metrics, anything task/model specific ...
data_loaders = build_loaders(cfg) # instantiate the data loaders
logging_params = buid_logging_params(cfg) # contains run_name, logging intervals, etc.

compute_params = get_defaults_compute()
compute_params.update(cfg) # optional
compute_params.use_amp = True # can also use this

trainer = Trainer(model, data_loaders, logging_params, compute_params)

trainer.run(epochs=5)
```




## Testing & Evaluation

