# scan2cap-with-transformers
- Our model is models/capnet_transformer.py
- Using models/meshed-memory-transformer/
- utils_meshed_memory_trainsformer


# Scan2Cap
This project builds upon the [Scan2Cap](https://github.com/daveredrum/Scan2Cap) architecture.

# Meshed Memory Transformer
We utilize the Meshed Memory Transformer architecture in this model. Therefore some files provided here originate from the [Meshed Memory Transformer](https://github.com/aimagelab/meshed-memory-transformer).

The license is provided in `LICENSE_Meshed_Memory_Transformer.md`
# Data
Please refer to [Scan2Cap](https://github.com/daveredrum/Scan2Cap) in order to get the datasets.

# Setup
Please also follow all setup steps provided in [Scan2Cap](https://github.com/daveredrum/Scan2Cap) for our project. Please make sure to install our `requirements.txt` with `pip install -r requirements.txt`

Additionally you will need to insert `create_scanrefer_filtered_train_small.py` into you `/data` folder and execute it.
- This will create a small subset of the training data for evaluation on items from the training set. This does not affect evaluation on items from the validation set.
- If you wish to use the whole training set during evaluation, just copy `ScanrRefer_filtered_train.json` to  `ScanrRefer_filtered_train_small.json` in your `/data` folder.

# Training our model
To reproduce our S2C-MMT results please first train the model for 50 epochs.
```bash
    python scripts/train.py --use_multiview --use_normal --use_orientation --use_relation --num_graph_steps 2 --num_locals 10 --batch_size=18 --epoch=50 --lr=0.001 --val_step=2000 --wd=0.0001 --transformer_dropout=0.1 --attention_module_memory_slots=20 --d_model=192 --transformer_d_ff=1024 --no_beam_search --transformer_d_k=32 --transformer_d_v=32 --no_encoder
```
As a next step train the model further for 5 epochs on following settings
```bash
    python scripts/train.py --use_multiview --use_normal --use_orientation --use_relation --num_graph_steps 2 --num_locals 10 --batch_size=18 --epoch=50 --lr=0.0001 --val_step=100 --wd=1e-4 --transformer_dropout=0.1 --attention_module_memory_slots=20 --d_model=192 --transformer_d_ff=1024 --no_beam_search --transformer_d_k=32 --transformer_d_v=32 --use_checkpoint=<model_name> --no_encoder --load_best
```
You will find your trained model in `outputs/<model_name>`.
# Reproducing results
We provide a pretrained model in `outputs/s2c_mmt` to reproduce the results. You can also use the steps mentioned in "Training our model" to train the model yourself.

Evaluating caption performance at 0.5IoU
```bash
 python scripts/eval.py --use_multiview --use_normal --use_relation --num_graph_steps 2 --num_locals 10 --batch_size=8 --transformer_dropout=0 --attention_module_memory_slots=20 --d_model=192 --transformer_d_ff=1024 --transformer_d_k=32 --transformer_d_v=32 --folder <model_name> --min_iou=0.5 --eval_caption --beam_size 2 --no_encoder 
```
Evaluating caption performance at 0.25IoU
```bash
 python scripts/eval.py --use_multiview --use_normal --use_relation --num_graph_steps 2 --num_locals 10 --batch_size=8 --transformer_dropout=0 --attention_module_memory_slots=20 --d_model=192 --transformer_d_ff=1024 --transformer_d_k=32 --transformer_d_v=32 --folder <model_name> --min_iou=0.25 --eval_caption --beam_size 2 --no_encoder 
```
Evaluating detection performance:
```bash
 python scripts/eval.py --use_multiview --use_normal --use_relation --num_graph_steps 2 --num_locals 10 --batch_size=8 --transformer_dropout=0 --attention_module_memory_slots=20 --d_model=192 --transformer_d_ff=1024 --transformer_d_k=32 --transformer_d_v=32 --folder <model_name> --eval_detection --no_beam_search --no_encoder 
```

# License
Scan2Cap is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License.

Meshed-Memory Transformer follwos BSD-3-Clause License.

Please refer to [Scan2Cap](https://github.com/daveredrum/Scan2Cap) and [Meshed Memory Transformer](https://github.com/aimagelab/meshed-memory-transformer) for licensing