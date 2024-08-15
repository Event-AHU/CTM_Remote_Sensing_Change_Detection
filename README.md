# CTM_RSCD
**Treat Stillness with Movement: Remote Sensing Change Detection via Coarse-grained Temporal Foregrounds Mining**, Zitian Wang†, Xixi Wang†, Jingtao Jiang, Lan Chen*, Xiao Wang, and Bo Jiang 
[[PDF]()]



## Abstract 
Current works focus on addressing the remote sensing change detection task using bi-temporal images. Although good performance can be achieved, however, seldom of they consider the motion cues which may also be vital. In this work, we revisit the widely adopted bi-temporal images-based framework and propose a novel Coarse-grained Temporal Mining Augmented (CTMA) framework. To be specific, given the bi-temporal images, we first transform them into a video using interpolation operations. Then, a set of temporal encoders is adopted to extract the motion features from the obtained video for coarse-grained changed region prediction. We also extract the motion features as an additional output to aggregate with the spatial features. Meanwhile, we feed the input image pairs into the ResNet to get the different features and also the spatial blocks for fine-grained feature learning. More importantly, we segment the coarse-grained changed regions and integrate them into the decoder blocks for final changed prediction. Extensive experiments conducted on multiple benchmark datasets fully validated the effectiveness of our proposed framework for remote sensing image change detection. 

<p align="center">
  <img src="https://github.com/Event-AHU/CTM_Remote_Sensing_Change_Detection/blob/main/figure/framework.jpg" alt="framework" width="700"/>
</p>


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
You can find the training script **train.py**. You can run the script file by python **train.py** in the command environment.

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


## Experiments 

<p align="center">
  <img src="https://github.com/Event-AHU/CTM_Remote_Sensing_Change_Detection/blob/main/figure/Feat_VIS1.jpg" alt="framework" width="700"/>
</p>

<p align="center">
  <img src="https://github.com/Event-AHU/CTM_Remote_Sensing_Change_Detection/blob/main/figure/Feat_VIS3.jpg" alt="framework" width="700"/>
</p>

<p align="center">
  <img src="https://github.com/Event-AHU/CTM_Remote_Sensing_Change_Detection/blob/main/figure/benchmark.jpeg" alt="framework" width="850"/>
</p>



## License
Code is released for **non-commercial** and **research purposes only**. For commercial purposes, please contact the authors.



## Citation
If you use this code for your research, please cite our paper:

```
@article{wang2024CTM,
  title={Treat Stillness with Movement: Remote Sensing Change Detection via Coarse-grained Temporal Foregrounds Mining},
  author={Zitian Wang, Xixi Wang, Jingtao Jiang, Lan Chen, Xiao Wang, and Bo Jiang},
  journal={},
  year={2024}
}
```

If you have any issues with this work, feel free to leave an issue for discussion. 


