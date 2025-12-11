from __future__ import annotations


print("Loading MAE preprocessor...")
print("Importing necessary modules...")
from nucli_train.data_management.builders import build_data

from nucli_train.training import Trainer
import numpy as np
import yaml
from nucli_train.preprocess.preprocessor import PreprocessorBlosc2
print("Importing necessary modules completed.")

print("\nLoading the MAE preprocessor...")

class MAEPreprocessor(PreprocessorBlosc2): 
    def __init__(self, **kwargs): 
        super().__init__(**kwargs)
    
    def exclude_condition(self, nifti_filename):
        return nifti_filename.endswith("0000.nii.gz") or ('u.' in nifti_filename)
    
    def identify_tracer(self, nifti_filename):
        if "fdg" in nifti_filename:
            return "fdg"
        elif "psma" in nifti_filename:
            return "psma"
        return "unknown"
        
    
    def identify_center(self, nifti_filename):
        return "autopet_center"
    
patch_size = (128, 128, 128)

kwargs = {
    "dataset_name": "autopet_2024",
    "dataset_path": "/home/interns_nuclivision_com/aws_backup/adam/autopet_2024",
    "exp_name": "MAE_ConvNeXt_adam_experiment",
    "nifti_input_rootdir": "/home/interns_nuclivision_com/aws_backup/adam/autopet_2024/imagesTr",
    "nifti_target_rootdir": None,
    "percentage_dataset": 0.1,
    "train_val_percentage": 0.8,
    "spacing": (2.0, 2.0, 3.0),
    "batch_size_train": 2,
    "batch_size_val": 2,
    "num_workers_train": 2,
    "num_workers_val": 2,
    "global_eval_interval": 10,
    "patch_size": patch_size,
    "shuffle_pick": True,
    "validation_evaluator": "save-preds-MAE", 
    "transformation": "mae_convnext_transform", 
    "transformation_args": {
        "volume_size": patch_size[0],
        "patch_size": 16,
        "mask_ratio": 0.6
    },
    "resample": True,
    "use_coords": True 
}


my_preprocessor = MAEPreprocessor(**kwargs)


