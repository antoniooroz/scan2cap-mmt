# scan2cap-with-transformers
- Our model is models/capnet_transformer.py
- Using models/meshed-memory-transformer/
- utils_meshed_memory_trainsformer


# Scan2Cap
This project is builds upon the Scan2Cap architecture
Original link: https://github.com/daveredrum/Scan2Cap

# Meshed Memory Transformer
Files provided in 
- /models/meshed_memory_transformer
- /utils_meshed_memory_transformer

originate from the Meshed Memory Transformer architecture. They are modified.

The license is provided in LICENSE_Meshed_Memory_Transformer.md

Original link: https://github.com/aimagelab/meshed-memory-transformer

# Data
Please refer to https://github.com/daveredrum/Scan2Cap in order to get the datasets.

# Setup
Please also follow all setup steps provided in https://github.com/daveredrum/Scan2Cap for our project.

Additionally you will need to insert create_scanrefer_filtered_train_small.py into you data folder and execute it.
- This will create a small subset of the training data for evaluation on items from the training set. This does not affect evaluation on items from the validation set.
- If you wish to use the whole training set during evaluation, just copy ScanrRefer_filtered_train.json to  ScanrRefer_filtered_train_small.json in your data folder.

# License
Scan2Cap is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License.

Meshed-Memory Transformer follwos BSD-3-Clause License.

Please refer to https://github.com/daveredrum/Scan2Cap and https://github.com/aimagelab/meshed-memory-transformer for licensing