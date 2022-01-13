# scan2cap-with-transformers
- Our model is models/capnet_transformer.py
- Using models/meshed-memory-transformer/

# Visualize

To visualize, run the following command:
```shell
python scripts/visualize.py
```
with arguments
```shell
--folder <folder-name-of-model-in-outputs-folder> --scene_id <folder-name-of-scene>
<parameters-used-to-call-training or eval>
```

example:
```shell
python scripts/visualize.py --folder 2022-01-13_13-23-25 --scene_id scene0000_00 --use_normal --use_topdown --use_relation --use_orientation --num_graph_steps 2 --num_proposals 32 --num_locals 10 --batch_size 2 
```

Results are stored in  `outputs/<model>/vis` and can be visualized with meshlab (available in snap).
