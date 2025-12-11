

## Models vs Networks

**nucli-train** has both *networks* and *models*; but what's the difference between them? A *model* is a broader term, encapsulating how the task at hand is formulated and optimized. 

For example, different denoising models exist:

- Training a single network based on some loss function between target and prediction
- Training a conditional diffusion model and performing inference by conditioning on the low-count image.
- CycleGAN, Pix2Pix, ...

Importantly, all these different methods (or models) are agnostic towards the specific neural network which is being optimized. Each of these methods can be used regardless if the network optimized is a U-Net, ViT, ResNet, ...

To reflect this in **nucli-train**, the *model* class exists. It allows defining training schemes, discriminators, ... agnostic to the network architecture used. In short, a *model* expresses the optimization objective and how network(s) are optimized, being agnostic to 
- Network architecture
- All data preprocessing and loading, this includes any resampling or normalization that happened *prior* to the training run
- Any deterministic processing of the raw in- and output at train- & test time, also referred to as data representation. 

Instead, a *model* is expected to have the following methods:

### train_step


### eval and train


### get_optimizers

### weights_to_save, get_params


### predict

### train_step


### validation_step


## Defining and returning losses
For a simple training scenario (one network) `nucli-train` expects the model's `train_step` to return  a dictionary containing key 'loss', which is where a backward call will be called, and other arguments representing any components of the loss which need to be logged.

Currently, we support multiple losses, in `train_step`, simply use yield each time a loss is calculated. Be careful! After yield the corresponding model parameters will be changed. The `get_optimizers`method should return the optimizer to be used for each loss. Right now, multiple optimizers per loss are not supported.