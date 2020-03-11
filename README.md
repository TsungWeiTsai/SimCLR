# SimCLR

Unofficial Pytorch Implementation of "A Simple Framework for Contrastive Learning of Visual Representations ([Paper](https://arxiv.org/abs/2002.05709))"

### Train on CIFAR 

Currently, we support training with LARS, amp or neither of them. The commands are as follows:

- LARS
```
CUDA_VISIBLE_DEVICES=0,1 python train_simclr.py \
 --LARS --nce_t 0.5 --epoch 1000 --learning_rate 1.0 --lr_warmup 10 --batch_size 256  --num_workers 8 \ 
 --contrastive_model simclr --aug simple --weight_decay 1e-6 --model resnet50_cifar --dataset cifar \
 --model_path ./model_save --tb_path ./tensorboard --data_folder ./  
```

- amp
```
CUDA_VISIBLE_DEVICES=0,1 python train_simclr.py \
 --amp --nce_t 0.5 --epoch 1000 --learning_rate 1.0 --lr_warmup 10 --batch_size 256  --num_workers 8 \
 --contrastive_model simclr --aug simple --weight_decay 1e-6 --model resnet50_cifar --dataset cifar \
 --model_path ./model_save --tb_path ./tensorboard --data_folder ./  
```

- Normal training
```
CUDA_VISIBLE_DEVICES=0,1 python train_simclr.py \
 --nce_t 0.5 --epoch 1000 --learning_rate 1.0 --lr_warmup 10 --batch_size 256  --num_workers 8 \
 --contrastive_model simclr --aug simple --weight_decay 1e-6 --model resnet50_cifar --dataset cifar \
 --model_path ./model_save --tb_path ./tensorboard --data_folder ./  
```

### Installation

- python 3.6+
- PyTorch 1.0+
- CUDA 10.0
- torchlars
- tensorboard_logger
- scikit-image
- apex (for amp)

## Notice

The implementation may be slightly different from the official implementation. For instance, we use the shuffle BN ([Paper](https://arxiv.org/abs/1911.05722) )instead of the Global BN trick. 
In addition, currently, the code is tested in CIFAR-10 only. We can attain around 90% accuracy after training 1000 epochs. 

If you should find any mistakes in the code or better hyper-parameter settings that achieves higher accuracy, please kindly let us know or submit the pull request. Thank you.


## Acknowledgements

The implementation heavily relies on the following repos:
1. CMC: Contrastive Multiview Coding ([Paper](http://arxiv.org/abs/1906.05849), [Code](https://github.com/HobbitLong/CMC))
2. InsDis: Unsupervised Feature Learning via Non-Parametric Instance-level Discrimination ([Paper](https://arxiv.org/abs/1805.01978), [Code](https://github.com/zhirongw/lemniscate.pytorch))
3. Unsupervised Embedding Learning via Invariant and Spreading Instance Feature ([Paper](https://arxiv.org/abs/1904.03436), [Code](https://github.com/mangye16/Unsupervised_Embedding_Learning))
4. SimCLR (official tensorflow implementation, [Code](https://github.com/google-research/simclr))
