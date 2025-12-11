

## Preparing your dataset
While dataloaders are supported for nifti files, [they are not recommended](https://www.notion.so/nuclivision/3D-Volume-sampling-2070ca36bdaa80b4a2c1f3bb129e6e82?source=copy_link). 

Instead, **nucli-train** offers dataset preparation & loading with two different methods:
- Save all ROIs from the dataset in a single .npy file
- Preprocess and save the entire dataset in HDF5 format

Choosing between the two will depend on your experiment's purpose & scale. See the Notion docs for guidelines and examples.

To preprocess and save a dataset, ensure your dataset follows the official Nuclivision long-term storage format, and run

`
nucli-train preprocess_dataset path/to/data_cfg.yaml
`

where the data_cfg.yaml looks like:

```python
train:
        dataset:
                center:
                        tracer:
                                inputs : [ct, 25pc] # just an example
                                targets: [100pc, organs] # just an example
                                patients : int | list
        ...
val:
        dataset:
                center:
                        tracer:
                                inputs : [ct, 25pc] # just an example
                                targets: [100pc, organs] # just an example
                                patients : int | list
                                resample_to : str #optional
                                target_spacing : [float, float, float] # optional

target_spacing :[float, float, float] # optional; will override tracer level arguments
patch_size : [float, float, float] # if one of three is -1, save 2D slices
src_dir : str # base directory where your datasets in the official long-term storage NiFTI format are located
save_dir : str # directory where processed dataset is saved
patch_coords : str # optional
```

Target spacing can be specified for all data, or per center/tracer. In PET-to-PET image translation, you can often pass a single target-spacing (or none at all). 

In more advanced use-cases (e.g CT), adding `resample_to : modality` will resample to the target modality (e.g 100pc, CT), being aware of image affines + spacing.

Finally, you can also specify `target_spacing` for a single center/tracer, in scenarios where a few scanners have similar voxel spacing and one deviates significantly. 

In self-supervised learning experiments, an empty list can be passed to targets. 

When saving in Blosc2 format, we use nnUNet's block size heuristics. To save all samples in a single .npy file, pass an additional argument `--patch_coords your_coords.npy` to your `preprocess_dataset`call

#### Working with multiple doses
Sometimes you can have multiple inputs/targets pairs, for example when denoising both 25% and 50% of standard-count. In those cases, name each pair (this name will be displayed in MLFlow validation metrics), and add each seperately to the tracer's dictionary:

```python
tracer:
        pair1:
                inputs : [ct, 10pc]
                targets: [organs, 10pc]
                patients : int | list

        pair2:
                inputs: [ct, 50pc]
                targets : [organs, 50pc]
                patients : int | list
```
The script will use the same set of patients for all pairs. When you want separate sets of patients for each pair, a workaround can be to generate your list of patients per pair beforehand.

## Generating patch coordinates
When saving all samples in a `.npy` file, the `preprocess_dataset` command requires an additional argument pointing to the file containing patch locations. **Nucli-train** has a seperate tool to generate these. Its logic is as follows:
1. Calculate suitable ROIs based on a heuristic. 
2. This list is often too large to save all patch candidates. A maximum overlap ratio is used to remove redundant patch locations.

Because generating target patch candidates is a complex topic which is not yet optimized, we do not offer a plug-and-play script. Instead, we provide a few utility functions which require a ROI finding function as input. 

For example, the following script can generate a list of ROI coordinates.
```python
from nucli_train.data_management import 


```