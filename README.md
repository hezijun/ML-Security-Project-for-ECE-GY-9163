# ML-Security-Project-for-ECE-GY-9163
```bash
├── data
    ├── Multi-trigger Multi-target
        └── eyebrows_poisoned_data.h5
        └── lipstick_poisoned_data.h5
        └── sunglasses_poisoned_data.h5
    └── anonymous_1_poisoned_data.h5
    └── clean_test_data.h5
    └── clean_validation_data.h5
    └── sunglasses_poisoned_data.h5
├── models
    └── sunglasses_bd_net.h5
    └── sunglasses_new_model.h5
    └── multi_trigger_multi_target_bd_net.h5
    └── multi_trigger_multi_target_new_model.h5
    └── anonymous_1_bd_net.h5
    └── anonymous_1_new_model.h5
    └── anonymous_2_bd_net.h5
    └── anonymous_2_new_model.h5
├── Fine-pruning and Evaluation
    └── sunglasses_bd_net.ipynb
    └── multi_trigger_multi_target_bd_net.ipynb
    └── anonymous_1_bd_net.ipynb
    └── anonymous_2_bd_net.ipynb
└── evaluation script
    └── eval_anonymous_1.py
    └── eval_anonymous_2.py
    └── eval_multi_trigger_multi_target.py
    └── sunglasses_bd_net.py
```

## I. Group Member
   - Zijun He
   - Xinyi Zhao
   - Zihan Li

## II. Dependencies
   - Python
   - Keras 
   - Numpy 
   - tensorflow_model_optimization
   - H5py 
   - Colab
   - matplotlib
   - sys
   
## III. Description:
   - [CSAW-HackML-2020](https://github.com/csaw-hackml/CSAW-HackML-2020)
   - data can be found in given google drive link
   - models contain 4 badnet models and 4 fine-pruning repaired models
   - Fine-pruning and Evaluation contain the training and evalution of each badnet model given
   - evaluation script contain one eval.py for large dataset evaluation, and four eval.py scripts, each corresponding to one of the four BadNets provided.

## IIII. Evaluating Operation
   1. To evaluate the fine-purning model with giving data, execute eval.py by running:
    python3 eval.py <testing data directory> <new model directory> <corresponding bad net model directory> .
    E.g., python3 eval.py data/testing_data.h5 models/sunglasses_new_model.h5 models/sunglasses_bd_net.h5
   2. To evaluate the repaired network with given input of YouTube Face image by running the corresponding .py by running:
    E.g., python3 eval_anonymous_2.py data/test_image.jpg
