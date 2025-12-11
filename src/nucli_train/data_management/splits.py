from __future__ import annotations

import copy
import os
import random


import yaml



def create_splits(cfg):


    data_dir = cfg['data_dir']

    data_cfg = cfg['data']
    train_set = {}
    val_set = {}
    test_set = {}
    split_dict = {'data_dir' : data_dir, 'train' : {}, 'val' : {}, 'test' : {}}
    for center, tracers in data_cfg.items():
        train_set[center] = {}
        val_set[center] = {}
        test_set[center] = {}
        for tracer, splits in tracers.items():
            train_split_tracer = {}

            patients = os.listdir(os.path.join(data_dir, center, tracer, '100pc'))



            # No patients in the train set multiple times (at different doses)
            for dose_id, dose in enumerate(splits["doses"]):
                if int(splits["train_sizes"][dose_id]):
                    train_ids = random.sample(range(len(patients)), int(splits["train_sizes"][dose_id]))

                    dose_train = [patients[i] for i in train_ids]
                    train_split_tracer[dose] = copy.deepcopy(dose_train)

                    patients = [patients[i] for i in range(len(patients)) if i not in train_ids]

            if train_split_tracer:
                train_set[center][tracer] = train_split_tracer


            val_ids = random.sample(range(len(patients)), int(splits["val_size"]))
            val_patient_list = [patients[i] for i in val_ids]
            val_set[center][tracer] = {'doses': splits['doses'], 'patients' : copy.deepcopy(val_patient_list)}

            test_patient_list = [patients[i] for i in range(len(patients)) if i not in val_ids]
            test_set[center][tracer] = copy.deepcopy(test_patient_list)
        
        if not train_set[center]:
            del train_set[center]

    split_dict['val'] = copy.deepcopy(val_set)
    split_dict['test'] = copy.deepcopy(test_set)
    split_dict['train'] = copy.deepcopy(train_set)



    return split_dict
