# Federated Learning

This is the code for the paper [Leveraging Foundation Models for Multi-modal Federated Learning with Incomplete Modality]

Note: The scripts will be slow without the implementation of parallel computing. 

## Abstract

Federated learning (FL) has obtained tremendous progress in providing collaborative training solutions for distributed data silos with privacy guarantees. While few of the existing works explore a more realistic scenario where the clients hold multiple data modalities. In this paper, we aim to solve two novel challenges in multi-modal federated learning(MFL), effective federated representation learning and modality missing, i.e. the clients may lose part of the modalities in their local data sets. In order to tackle these problems, we proposed a novel multi-modal federated learning method, Federated Multi-modal contrastiVe training with Pre-trained completion (FedMVP), which integrated the large-scale pre-trained models to enhance the federated training. In the proposed FedMVP framework, each client deploys a large-scale pre-trained model with frozen parameters for modality completion and representation knowledge transfer, enabling efficient and robust local training. On the server side, we utilize generated data to uniformly measure the representation similarity among the uploaded client models and construct a graph perspective to aggregate them according to their importance in the system. We demonstrate the model achieves superior performance over two image-text classification datasets with robustness to the performance degradation caused by modality missing.


## Requirements
python>=3.8

pytorch>=1.13

numpy>=1.21

matplotlib>=3.5

transformers>=4.25

spacy>=3.3

timm>=0.6


## Run

Parameter setting:
Modify [/utils/options.py](utils/options.py)

Federated learning:
> python [main_fed.py](main_fed.py)

See the arguments in [options.py](utils/options.py). 

For example:
> python main_fed.py --dataset cub --model fedmvp --epochs 300 --missing_ratio 0.3 --gpu 0  

## Datasets

#### [Caltech-UCSD Birds-200-2011](https://www.vision.caltech.edu/datasets/cub_200_2011/) dataset

- Download the data [here](https://www.vision.caltech.edu/datasets/cub_200_2011/) and extracted it to /data/cub

#### [Oxford 102 Flower](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/) dataset

- Download the data [here](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/) and extracted it to /data/flower



<!-- ## Results
### MNIST
Results are shown in Table 1 and Table 2, with the parameters C=0.1, B=10, E=5.

Table 1. results of 10 epochs training with the learning rate of 0.01

| Model     | Acc. of IID | Acc. of Non-IID|
| -----     | -----       | ----           |
| FedAVG-MLP|  94.57%     | 70.44%         |
| FedAVG-CNN|  96.59%     | 77.72%         |

Table 2. results of 50 epochs training with the learning rate of 0.01

| Model     | Acc. of IID | Acc. of Non-IID|
| -----     | -----       | ----           |
| FedAVG |      |         |
| FedAVG |      |         | -->


<!-- ## Ackonwledgements
Acknowledgements give to .

## References


## Cite As
 -->


