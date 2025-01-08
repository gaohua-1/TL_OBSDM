# Semantic Decoupling and Transfer Learning for Enhanced Small Object Detection
***
# **our approach**
.<div align=center>![b828105ddc6d1919d0a7ca26a9a34cd](https://github.com/user-attachments/assets/20fe09b7-38c3-49cd-bd29-b852c7ca8015)</div> 
***
.<div align=center>![56e6d30140568a035e7ab8b1278c2ac](https://github.com/user-attachments/assets/6110a8de-1e7e-4261-9178-938c6eaeb106)</div> 
***
.<div align=center>![399644586-d48df763-f858-4d58-9cb3-18fb1a35d1fd](https://github.com/user-attachments/assets/3ab4e6e2-6c91-4a3a-87ef-bb694f451dfb)</div> 
***
.<div align=center>![8ddea38da8c2c6cadf92dd23045aab3](https://github.com/user-attachments/assets/5723a99c-be51-492f-9b84-497a9a3a391f)</div> 
***

## **Requirements**  
```python
matplotlib>=3.2.2
numpy>=1.18.5,<1.24.0
opencv-python>=4.1.1
Pillow>=7.1.2
PyYAML>=5.3.1
requests>=2.23.0
scipy>=1.4.1
torch>=1.7.1,!=1.12.0
torchvision>=0.8.2,!=0.13.0
tqdm>=4.41.0
protobuf<4.21.3
Tensorboard
kornia==0.4.1
sklearn
```  
You can also use this command  
```python
pip install -r requirements.txt
```
***
## **OBSDM**  
### **How to use ?**    
Just put these project's items in yolov7 and you're ready to go.  
1. Download the dataset  
2. Edit the dataset.yaml,  in config files in ./data/  
3. Download our training models
***
### **Core File**  
cfg/training/yolov7-resnet18-cross.yaml  
Selfdetsim/logdetsim/imagenet100/ckpt-100.pth  
models/  
train.py
***
   
### **How to train ?**  
```python
python train.py --workers$  --device$  --sync-bn --batch-size$  --img $ --data data.yaml path --cfg model.yaml path --weights aug --mopath ckpt-100.pth  --name&  --hyp data/hyp.scratch.p5.yaml --epochs$
```
***

### **How to test ?**   
```python
python test.py --weights weights.pt path --data data.yaml path --batch-size$ --device$
```
***
### **Using our trained model**  
You can download our trained object detction model:run/train/yolov7-resnet18rail-obsdm/weights/weight
***
## **Transfer learning**  
### **How to pretrain**  
```python
python pretrain.py \
    --logdir ./logs/imagenet100 \
    --framework Detsim \
    --dataset imagenet100 \
    --datadir DATADIR \
    --batch-size 128 \
    --max-epochs 100 \
    --model resnet18 \
    --base-lr 0.05 --wd 1e-4 \
    --ckpt-freq 50 --eval-freq 50 \
    --ss-crop 0.5 --ss-color 0.5 \
--num-workers 16 --distributed
```
***
### **Core File**   
pretain.py  
train.py  
***
### **Using our trained model** 
You can download our pretrained model: Selfdetsim/logdetsim/imagenet100/ckpt-100.pth
***
## **Dataset**  
Voc:  http://host.robots.ox.ac.uk/pascal/VOC/index.html  
Tranficlight: https://hci.iwr.uni-heidelberg.de/content/bosch-small-traffic-lights-dataset  
Rail worker:  https://pan.baidu.com/s/1R54k-tvApwXgfNN2d3HZag (VNU9)  
***

# **Citing TL-OBSDM** 
@article{Liu25TLSemantic Decoupling,  
  title={Semantic Decoupling and Transfer Learning for Enhanced Small Object Detection},  
  author={Gaohua Liu and JinghaoZhang and Junhuan Li and Shuxia Yan and Xiangyu Kong and Rui Liu and Yueyang Li},  
  year = {2025},  
  volume={},  
  number={},  
  pages={-},  
  journal={The Visual Computer}  
}  
