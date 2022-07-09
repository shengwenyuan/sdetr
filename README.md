

## SDETR :Stacked DETR
We present a new model named Stacked-DETR(SDETR), which inherits the main ideas in canonical DETR. We improve DETR in two directions: simplifying the cost of training and introducing the stacked architecture to enhance the performance.
To the former, we focus on the inside of the Attention block and propose the QKVA grid, a new perspective	to describe the process of attention. By this, we can step further on how Attention works for image problems and the effect of multi-head. These two ideas contribute the design of single-head encoder-layer.
To the latter, SDETR reaches great improvement to DETR. Especially to the performance on small objects, SDETR achieves better results to the optimized Faster R-CNN baseline, which was a shortcoming in DETR.

## About the code
Our designs are based on the code of DETR. To preclude all the influences beyond our changes, we leave every other file, which may never be used in our model (like Detectron2 wrapper).


## Models
Because of limited resources, we implement most of experiments based on tiny-scaled input version and they can not be a convincing results to prove any performance. Here we trained the full-scaled input version with 150 epochs.  
Download url: <https://pan.baidu.com/s/1nVBXlQVYrZ1o88JV2nb6uQ> Token: 0000  
We will provide a new version with 500 epochs in future. 


## Usage
There are no extra compiled components in DETR and package dependencies are minimal, so the code is very simple to use.
First, clone the repository locally:
```
git clone https://github.com/shengwenyuan/sdetr
```
Then, install requirements:
```
pip install -r requirements.txt
```

### Data preparation
Same as DETR:  
Download and extract COCO 2017 train and val images with annotations from
[http://cocodataset.org](http://cocodataset.org/#download).
We expect the directory structure to be the following:
```
path/to/coco/
  annotations/  # annotation json files
  train2017/    # train images
  val2017/      # val images
```

## Training
To train SDETR on a single node with 4 RTX3090 gpus for 150 epochs run:
```
python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py --epochs=150 --lr_drop 100 --coco_path /path/to/coco 
```
A single epoch takes 56 minutes, so 150 epoch training
takes around 6 days on a single machine with 4 RXT3090 cards.
We provide the capatured stdio in file *F3-13-June241619*.

Same as DETR:  
We train DETR with AdamW setting learning rate in the transformer to 1e-4 and 1e-5 in the backbone.
Horizontal flips, scales and crops are used for augmentation.
Images are rescaled to have min size 800 and max size 1333.
The transformer is trained with dropout of 0.1, and the whole model is trained with grad clip of 0.1.


