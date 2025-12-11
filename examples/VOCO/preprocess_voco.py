from __future__ import annotations


print("Loading VOCO preprocessor...")
print("Importing necessary modules...")
from nucli_train.data_management.builders import build_data
from nucli_train.models.builders import build_model, MODEL_REGISTRY, MODEL_BUILDERS_REGISTRY
from nucli_train.training import Trainer
import voco_transform as voto
import numpy as np
import yaml
from nucli_train.preprocess.preprocessor import PreprocessorBlosc2
print("Importing necessary modules completed.")

print("\nLoading the VOCO preprocessor...")

class VoCoPreprocessor(PreprocessorBlosc2): 
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
    


kwargs = {
    "dataset_name": "autopet_2024",
    "dataset_path": "/home/interns_nuclivision_com/aws_backup/adam/autopet_2024",
    "exp_name": "VOCO_adam_experiment",
    "nifti_input_rootdir": "/home/interns_nuclivision_com/aws_backup/adam/autopet_2024/imagesTr",
    "nifti_target_rootdir": None,
    "percentage_dataset": 0.1,
    "train_val_percentage": 0.8,
    "spacing": (4.0, 4.0, 4.0),
    "batch_size_train": 8,
    "batch_size_val": 8,
    "num_workers_train": 2,
    "num_workers_val": 2,
    "global_eval_interval": 10,
    "patch_size": (128, 128, 128),
    "shuffle_pick": True,
    "validation_evaluator": "save-preds-VOCO", 
    "transformation": "voco_transform", 
    "transformation_args": {
        "voco_base_crop_count": (2, 2, 2), 
        "voco_crop_size": (64, 64, 64),
        "voco_target_crop_count": 4
    },
    "resample": True,
    "use_coords": False  
}


my_preprocessor = VoCoPreprocessor(**kwargs)



