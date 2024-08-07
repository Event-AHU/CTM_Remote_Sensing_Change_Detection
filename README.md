# CTM_RSCD
Treat Stillness with Movement: Remote Sensing Change Detection via Coarse-grained Temporal Foregrounds Mining

## Requirements
```
Python 3.7
pytorch 1.11.0
einops  0.6.0
torch-scatter 2.0.9
scipy 1.7.3
matplotlib  3.5.3
```
## Train
You can find the training script train.py. You can run the script file by python train.py in the command environment.

You can train on the svcd train set with the following command:
```
python -W ignore train.py train --exp_config ../configs/svcd/config_svcd_p2v.yaml --resume ../exp/svcd/weights/checkpoint_latest_p2v.pth
```
The same goes for the other datasets.
## Evaluate
After training, we can use the best saved model for evaluation.

You can evaluate on the svcd test set with the following command:
```
python -W ignore train.py eval --exp_config ../configs/svcd/config_svcd_p2v.yaml --resume ../exp/svcd/weights/model_best_p2v.pth --save_on --subset test
```
The same goes for the other datasets.
