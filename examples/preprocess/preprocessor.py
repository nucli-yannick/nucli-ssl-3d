from __future__ import annotations

"""
The goal of this script is to: 
    - preprocess a .nifti dataset to a .b2nd dataset
    - computing the potential centers
    - create the required .yaml files
"""


from typing import Tuple
from nifti_blosc2_transformer import Blosc2Compressor
import os 
import numpy as np
from potential_centers import process_one_case

import nibabel as nib

from nucli_train.data_management.create_dataset import save_blosc

import shutil
from ruamel.yaml import YAML
from ruamel.yaml.scalarstring import DoubleQuotedScalarString
from ruamel.yaml.comments import CommentedMap, CommentedSeq

yaml = YAML()
yaml.default_flow_style = False  # Dictionnaires indentés
yaml.indent(mapping=2, sequence=4, offset=2)



class PreprocessorBlosc2: 
    
    def __init__(
            self, 
            dataset_name,
            dataset_path,
            nifti_input_rootdir, 
            nifti_target_rootdir,
            percentage_dataset: float, # percentage of the dataset to use
            train_val_percentage: float, # percentage of the dataset to use for training
            spacing, 
            batch_size_train: int, 
            batch_size_val: int,
            num_workers_train: int,
            num_workers_val: int,
            global_eval_interval: int,
            validation_evaluator: str,
            input_name: str = "input",
            target_name: str = "target",
            patch_size: Tuple[int, int, int] = (64, 64, 64),
            shuffle_pick = False, 
            unique_pair = True

    ):

        self.dataset_name = dataset_name
        self.nifti_input_rootdir = nifti_input_rootdir
        self.nifti_target_rootdir = nifti_target_rootdir 
        self.spacing = spacing
        self.batch_size_train = batch_size_train
        self.batch_size_val = batch_size_val
        self.num_workers_train = num_workers_train
        self.num_workers_val = num_workers_val
        self.global_eval_interval = global_eval_interval
        self.patch_size = patch_size
        self.percentage_dataset = percentage_dataset
        self.train_val_percentage = train_val_percentage
        self.validation_evaluator = validation_evaluator
        self.dataset_path = dataset_path

        self.blosc2_compressor = Blosc2Compressor()

        list_potential_input_nifti = os.listdir(self.nifti_input_rootdir)
        if self.nifti_target_rootdir is not None:
            list_potential_target_nifti = os.listdir(self.nifti_target_rootdir)
            if len(list_potential_input_nifti) != len(list_potential_target_nifti):
                raise ValueError("The number of input and target NIfTI files must be the same.")

        self.list_potential_input_nifti = [
            nifti for nifti in list_potential_input_nifti if not self.exclude_condition(nifti)
        ]

        if self.nifti_target_rootdir is not None:
            self.list_potential_target_nifti = [
                nifti for (nifti, nifti_input) in zip(list_potential_target_nifti, list_potential_input_nifti) if not self.exclude_condition(nifti_input)
            ]
            if len(self.list_potential_input_nifti) != len(self.list_potential_target_nifti):
                raise ValueError("The number of input and target NIfTI files must be the same after filtering.")

        self.number_of_cases = int(len(self.list_potential_input_nifti) * self.percentage_dataset)

        if shuffle_pick: 
            self.chosen_indices = np.random.permutation(np.arange(len(list_potential_input_nifti)))[:self.number_of_cases]
        else: 
            self.chosen_indices = np.arange(self.number_of_cases)


        # indices used for self.list_potential_input_nifti and self.list_potential_target_nifti
        self.chosen_indices_train = self.chosen_indices[:int(self.number_of_cases * self.train_val_percentage)]
        self.chosen_indices_val = self.chosen_indices[int(self.number_of_cases * self.train_val_percentage):]

        # Creating the final lists of nifti files
        self.chosen_input_nifti_train = [self.list_potential_input_nifti[i] for i in self.chosen_indices_train]
        self.chosen_input_nifti_val = [self.list_potential_input_nifti[i] for i in self.chosen_indices_val]

        if self.nifti_target_rootdir is not None:
            self.chosen_target_nifti_train = [self.list_potential_target_nifti[i] for i in self.chosen_indices_train]
            self.chosen_target_nifti_val = [self.list_potential_target_nifti[i] for i in self.chosen_indices_val]
            if len(self.chosen_input_nifti_train) != len(self.chosen_target_nifti_train):
                raise ValueError("The number of input and target NIfTI files must be the same after filtering for training and validation.")

        # Identify the tracers
        self.train_tracer = [self.identify_tracer(self.list_potential_input_nifti[i]) for i in self.chosen_indices_train]
        self.val_tracer = [self.identify_tracer(self.list_potential_input_nifti[i]) for i in self.chosen_indices_val]

        # Identify the centers
        self.train_center = [self.identify_center(self.list_potential_input_nifti[i]) for i in self.chosen_indices_train]
        self.val_center = [self.identify_center(self.list_potential_input_nifti[i]) for i in self.chosen_indices_val]


        self.dico_train = {self.dataset_name: {}}
        self.dico_val = {self.dataset_name: {}}

        # Organize the paths
        self.input_train_dico = self.organize_paths(self.train_center, self.train_tracer, self.chosen_input_nifti_train)
        self.input_val_dico = self.organize_paths(self.val_center, self.val_tracer, self.chosen_input_nifti_val)

        if self.nifti_target_rootdir is not None:
            self.target_train_dico = self.organize_paths(self.train_center, self.train_tracer, self.chosen_target_nifti_train)
            self.target_val_dico = self.organize_paths(self.val_center, self.val_tracer, self.chosen_target_nifti_val)
        
        self.pairname = "full body pet scans"



        self.nucli_train_path = os.path.join(os.path.join(self.dataset_path, "nucli_train"), self.dataset_name)
        if not os.path.exists(self.nucli_train_path):
            os.makedirs(self.nucli_train_path)
        else: 
            shutil.rmtree(self.nucli_train_path)
            os.makedirs(self.nucli_train_path)

        self.b2ndrootdir = os.path.join(self.nucli_train_path, "blosc2_" + str(self.patch_size[0]) + "_" + str(self.patch_size[1]) + "_" + str(self.patch_size[2]))

        if not os.path.exists(self.b2ndrootdir):
            os.makedirs(self.b2ndrootdir)
        else: 
            shutil.rmtree(self.b2ndrootdir)
            os.makedirs(self.b2ndrootdir)

        self.b2ndrootdir_train = self.b2ndrootdir + "/training"
        self.b2ndrootdir_val = self.b2ndrootdir + "/validation"

        # Create the directories for training and validation
        if not os.path.exists(self.b2ndrootdir_train):
            os.makedirs(self.b2ndrootdir_train)
        else: 
            shutil.rmtree(self.b2ndrootdir_train)
            os.makedirs(self.b2ndrootdir_train)
            
        if not os.path.exists(self.b2ndrootdir_val):
            os.makedirs(self.b2ndrootdir_val)
        else: 
            shutil.rmtree(self.b2ndrootdir_val)
            os.makedirs(self.b2ndrootdir_val)

        # Create the directories

        for center, tracers in self.input_train_dico.items():
            for tracer, nifti_filenames in tracers.items():
                specific_dir = os.path.join(self.b2ndrootdir_train, self.dataset_name, center, tracer, "input")
                if not os.path.exists(specific_dir):
                    os.makedirs(specific_dir)
                else: 
                    shutil.rmtree(specific_dir)
                    os.makedirs(specific_dir)
                coords_dir = os.path.join(self.b2ndrootdir_train, self.dataset_name, center, tracer, "coords")
                if not os.path.exists(coords_dir):
                    os.makedirs(coords_dir)
                else: 
                    shutil.rmtree(coords_dir)
                    os.makedirs(coords_dir)
        
        for center, tracers in self.input_val_dico.items():
            for tracer, nifti_filenames in tracers.items():
                specific_dir = os.path.join(self.b2ndrootdir_val, self.dataset_name, center, tracer, "input")
                if not os.path.exists(specific_dir):
                    os.makedirs(specific_dir)
                else: 
                    shutil.rmtree(specific_dir)
                    os.makedirs(specific_dir)
                coords_dir = os.path.join(self.b2ndrootdir_val, self.dataset_name, center, tracer, "coords")
                if not os.path.exists(coords_dir):
                    os.makedirs(coords_dir)
                else: 
                    shutil.rmtree(coords_dir)
                    os.makedirs(coords_dir)
        
        if self.nifti_target_rootdir is not None:
            for center, tracers in self.target_train_dico.items():
                for tracer, nifti_filenames in tracers.items():
                    specific_dir = os.path.join(self.b2ndrootdir_train, self.dataset_name, center, tracer, "target")
                    if not os.path.exists(specific_dir):
                        os.makedirs(specific_dir)
                    else: 
                        shutil.rmtree(specific_dir)
                        os.makedirs(specific_dir)
                    coords_dir = os.path.join(self.b2ndrootdir_train, self.dataset_name, center, tracer, "coords")
                    if not os.path.exists(coords_dir):
                        os.makedirs(coords_dir)
                    else: 
                        shutil.rmtree(coords_dir)
                        os.makedirs(coords_dir)
                    
            for center, tracers in self.target_val_dico.items():
                for tracer, nifti_filenames in tracers.items():
                    specific_dir = os.path.join(self.b2ndrootdir_val, self.dataset_name, center, tracer, "target")
                    if not os.path.exists(specific_dir):
                        os.makedirs(specific_dir)
                    else: 
                        shutil.rmtree(specific_dir)
                        os.makedirs(specific_dir)
                    coords_dir = os.path.join(self.b2ndrootdir_val, self.dataset_name, center, tracer, "coords")
                    if not os.path.exists(coords_dir):
                        os.makedirs(coords_dir)
                    else: 
                        shutil.rmtree(coords_dir)
                        os.makedirs(coords_dir)
        
        # generate the yaml files
        self.create_config_files()
        
        # Compress the NIfTI files
        self.compress_and_coords(self.input_train_dico, self.b2ndrootdir_train, modality="input")
        self.compress_and_coords(self.input_val_dico, self.b2ndrootdir_val, modality="input")

        if self.nifti_target_rootdir is not None:
            self.compress_and_coords(self.target_train_dico, self.b2ndrootdir_train, modality="target")
            self.compress_and_coords(self.target_val_dico, self.b2ndrootdir_val, modality="target")

        


    def load_nifti(self, path): 
        img = nib.load(path)
        volume = img.get_fdata().astype(np.float16)
        return volume


        
    def compress_and_coords(self, dico, path, modality: str = "input"):
        """
        Compress the NIfTI files to .b2nd files
        """
        
        for center, tracers in dico.items():
            for tracer, nifti_filenames in tracers.items():
                for nifti_filename in nifti_filenames:
                    # Create the specific file
                    nifti_path = os.path.join(self.nifti_input_rootdir, nifti_filename)
                    specific_file_path_b2nd = os.path.join(path, self.dataset_name, center, tracer, modality, nifti_filename.replace('.nii.gz', ''))

                    volume = self.load_nifti(nifti_path)
                    
                    #self.blosc2_compressor.save_case(volume, self.patch_size, specific_file_path_b2nd)
                    save_blosc(specific_file_path_b2nd, volume, np.array(self.patch_size))
                    print(f"Compressed {nifti_filename} to {specific_file_path_b2nd}")
                    coords_path = os.path.join(path, self.dataset_name, center, tracer, "coords", nifti_filename.replace('.nii.gz', '.npy'))
                    process_one_case(volume, coords_path, self.patch_size)

                    print(f"saved coordinates to {coords_path}")
                    print("\n \n \n ")
                    

    def organize_paths(self, centers, tracers, nifti_filenames) -> dict:
        """
        Organizes the paths of the nifti files into a dictionary
        """
        organized_paths = {}
        for center, tracer, nifti_filename in zip(centers, tracers, nifti_filenames):
            if center not in organized_paths:
                organized_paths[center] = {}
            if tracer not in organized_paths[center]:
                organized_paths[center][tracer] = []
            organized_paths[center][tracer].append(nifti_filename)
        return organized_paths

    def create_config_files(self): 

        # Creation of the train config path
        train_config_path = os.path.join(self.nucli_train_path, "train.yaml")

        # Creation of the validations config paths
        val_config_path = os.path.join(self.nucli_train_path, "validation.yaml")

        # Creation of the main config path
        main_config_path = os.path.join(self.nucli_train_path, "main.yaml")

        # Writing the main config file
        self.write_main_yaml(main_config_path, train_config_path, val_config_path)
        # Writing the train config file
        self.write_train_yaml(train_config_path)
        # Writing the validation config file
        self.write_val_yaml(val_config_path)


    def write_main_yaml(self, main_config_path: str, train_config_path: str, val_config_path: str):
        """
        Writes the main YAML file for the dataset
        """
        data = {
        'train': {
            'type': 'blosc2_dataset',
            'params': {
                'config_path': train_config_path,
            },
            'batch_size': self.batch_size_train,
            'num_workers': self.num_workers_train,
        },
        'val': {
            'datasets': {
                'tracers': {
                    'type': 'blosc2_dataset',
                    'params': {
                        'config_path': val_config_path
                    },
                    'evaluators': [
                        self.validation_evaluator
                    ]
                }
            },
            'batch_size': self.batch_size_val,
            'num_workers': self.num_workers_val,
            'global_eval_interval': self.global_eval_interval
        }
        }

        with open(main_config_path, 'w') as f:
            yaml.dump(data, f)

    def write_train_yaml(self, train_config_path: str):
        data = {
            "data": {
                self.dataset_name: {

                }
            },
            "patch_size": self.patch_size,
            "src_dir": self.b2ndrootdir_train,
            "storage-type": "blosc2",
            "spacing": self.spacing
        }

        for center, tracers in self.input_train_dico.items():
            for tracer, nifti_filenames in tracers.items():
                if center not in data["data"][self.dataset_name]:
                    data["data"][self.dataset_name][center] = {}
                if tracer not in data["data"][self.dataset_name][center]:
                    data["data"][self.dataset_name][center][tracer] = {
                        'pairs': {self.pairname: {'inputs': ["input"], 'targets': ["target"] if self.nifti_target_rootdir is not None else []}},
                        'patients': 'all'
                        }
                    

        with open(train_config_path, 'w') as f:
            yaml.dump(data, f)
    
    def write_val_yaml(self, val_config_path: str):
        data = {
            "data": {
                self.dataset_name: {

                }
            },
            "patch_size": self.patch_size,
            "src_dir": self.b2ndrootdir_val,
            "storage-type": "blosc2",
            "spacing": self.spacing
        }

        for center, tracers in self.input_val_dico.items():
            for tracer, nifti_filenames in tracers.items():
                if center not in data["data"][self.dataset_name]:
                    data["data"][self.dataset_name][center] = {}
                if tracer not in data["data"][self.dataset_name][center]:
                    data["data"][self.dataset_name][center][tracer] = {
                        'pairs': {self.pairname: {'inputs': ["input"], 'targets': ["target"] if self.nifti_target_rootdir is not None else []}},
                        'patients': 'all'
                        }
                    
        
        with open(val_config_path, 'w') as f:
            yaml.dump(data, f)
                

    def exclude_condition(self, nifti_filename: str) -> bool: 
        """
        Has to be overwritten in the child class
        """
        return False

    def identify_tracer(self, nifti_filename: str) -> str:
        """
        Has to be overwritten in the child class
        """
        return "unknown"
    
    def identify_center(self, nifti_filename: str) -> str:
        """
        Has to be overwritten in the child class
        """
        return "unknown"
    

    def build_main_yaml(self, output_path: str):
        """
        Builds the main YAML file for the dataset
        """
        pass
        


if __name__ == "__main__":
    # Example usage
    preprocessor = PreprocessorBlosc2(
        "autopet_2024",
        "/home/ec2-user/data/autopet_2024",
        nifti_input_rootdir="/home/ec2-user/data/autopet_2024/imagesTr",
        nifti_target_rootdir=None,
        percentage_dataset=0.005,
        train_val_percentage=0.8,
        spacing=(1.0, 1.0, 1.0),
        batch_size_train=32,
        batch_size_val=16,
        num_workers_train=4,
        num_workers_val=2,
        global_eval_interval=1000,
        patch_size=(64, 64, 64),
        shuffle_pick=True, 
        validation_evaluator="evaluator_name", 
    )

    


    




        
        




    

    
