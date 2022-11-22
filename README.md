# Usage Example by Ryo(me)

## background
- CUDA is installed

```
$ nvcc -V
---------------
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2021 NVIDIA Corporation
Built on Mon_May__3_19:41:42_Pacific_Daylight_Time_2021
Cuda compilation tools, release 11.3, V11.3.109
Build cuda_11.3.r11.3/compiler.29920130_0
```

- anaconda is installed
```
$ conda info
---------------------------------------------------
     active environment : None
            shell level : 0
       user config file : C:\Users\island\.condarc
 populated config files : C:\Users\island\.condarc
          conda version : 22.9.0
    conda-build version : 3.23.1
         python version : 3.8.8.final.0
       virtual packages : __cuda=11.7=0
                          __win=0=0
                          __archspec=1=x86_64
       base environment : C:\Users\island\anaconda3  (writable)
      conda av data dir : C:\Users\island\anaconda3\etc\conda
  conda av metadata url : None
           channel URLs : https://repo.anaconda.com/pkgs/main/win-64
                          https://repo.anaconda.com/pkgs/main/noarch
                          https://repo.anaconda.com/pkgs/r/win-64
                          https://repo.anaconda.com/pkgs/r/noarch
                          https://repo.anaconda.com/pkgs/msys2/win-64
                          https://repo.anaconda.com/pkgs/msys2/noarch
          package cache : C:\Users\island\anaconda3\pkgs
                          C:\Users\island\.conda\pkgs
                          C:\Users\island\AppData\Local\conda\conda\pkgs
       envs directories : C:\Users\island\anaconda3\envs
                          C:\Users\island\.conda\envs
                          C:\Users\island\AppData\Local\conda\conda\envs
               platform : win-64
             user-agent : conda/22.9.0 requests/2.25.1 CPython/3.8.8 Windows/10 Windows/10.0.19041
          administrator : False
             netrc file : None
           offline mode : False
```

## Installation 

```
$ conda create -n unet 
$ conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
$ pip install -r requirements.txt
```
go to https://www.kaggle.com/competitions/carvana-image-masking-challenge/data, and intall below (Kaggle Signin reqired).

- `train_hq.zip`, extract contents to `data/imgs`
- `train_masks.zip`, extract contents to `data/masks`

### Problem #1

- To install Pythorch, below error was occured
```
$ conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
ref) https://pytorch.org/get-started/previous-versions/
---
Can't connect to HTTPS URL because the SSL module is not available.
```

### How to solve Problem #1

- To add `C:\Users\island\anaconda3\Library\bin` to PATH 


## Train

```
$ python train.py
----------
INFO: Using device cuda
INFO: Network:
        3 input channels
        2 output channels (classes)
        Transposed conv upscaling
INFO: Creating dataset with 5088 examples
wandb: Currently logged in as: anony-moose-446188. Use `wandb login --relogin` to force relogin
wandb: Tracking run with wandb version 0.13.5
wandb: Run data is saved locally in D:\git\Pytorch-UNet\wandb\run-20221122_093746-37zf8lx8
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run rare-pyramid-2
wandb:  View project at https://wandb.ai/anony-moose-446188/U-Net?apiKey=9376384096a26debc2c33745dfffa759f3ac9348
wandb:  View run at https://wandb.ai/anony-moose-446188/U-Net/runs/37zf8lx8?apiKey=9376384096a26debc2c33745dfffa759f3ac9348
wandb: WARNING Do NOT share these links with anyone. They can be used to claim your runs. (Ryo Commented -> It's OK.)
INFO: Starting training:
        Epochs:          5
        Batch size:      1
        Learning rate:   1e-05
        Training size:   4580
        Validation size: 508
        Checkpoints:     True
        Device:          cuda
        Images scaling:  0.5
        Mixed Precision: False

Epoch 1/5:  10%|████▋                                          | 458/4580 [05:02<42:38,  1.61img/s, loss (batch)=0.119]INFO: Validation Dice score: 0.9450088143348694
Epoch 1/5:  20%|█████████▏                                    | 916/4580 [11:51<38:03,  1.60img/s, loss (batch)=0.0302]INFO: Validation Dice score: 0.9728356599807739
Epoch 1/5:  30%|█████████████▌                               | 1374/4580 [18:38<33:30,  1.59img/s, loss (batch)=0.0224]INFO: Validation Dice score: 0.9644173979759216
Epoch 1/5:  40%|██████████████████                           | 1832/4580 [25:25<28:28,  1.61img/s, loss (batch)=0.0185]INFO: Validation Dice score: 0.9732168316841125
Epoch 1/5:  50%|██████████████████████▌                      | 2290/4580 [32:11<23:46,  1.61img/s, loss (batch)=0.0198]INFO: Validation Dice score: 0.9726505279541016
Epoch 1/5:  60%|███████████████████████████                  | 2748/4580 [39:00<19:21,  1.58img/s, loss (batch)=0.0155]INFO: Validation Dice score: 0.9734430313110352
Epoch 1/5:  70%|███████████████████████████████▍             | 3206/4580 [45:47<14:15,  1.61img/s, loss (batch)=0.0281]INFO: Validation Dice score: 0.9608033895492554
Epoch 1/5:  80%|████████████████████████████████████         | 3664/4580 [52:35<09:32,  1.60img/s, loss (batch)=0.0472]INFO: Validation Dice score: 0.9812467694282532
Epoch 1/5:  90%|████████████████████████████████████████▌    | 4122/4580 [59:23<04:46,  1.60img/s, loss (batch)=0.0115]INFO: Validation Dice score: 0.9834566116333008
Epoch 1/5: 100%|███████████████████████████████████████████| 4580/4580 [1:06:11<00:00,  1.59img/s, loss (batch)=0.0139]INFO: Validation Dice score: 0.9817652702331543
Epoch 1/5: 100%|███████████████████████████████████████████| 4580/4580 [1:08:13<00:00,  1.12img/s, loss (batch)=0.0139]
INFO: Checkpoint 1 saved!
Epoch 2/5:  10%|████▌                                         | 458/4580 [05:00<43:01,  1.60img/s, loss (batch)=0.0208]INFO: Validation Dice score: 0.9838627576828003
Epoch 2/5:  20%|█████████▏                                    | 916/4580 [11:49<38:21,  1.59img/s, loss (batch)=0.0303]INFO: Validation Dice score: 0.9871749877929688
Epoch 2/5:  30%|█████████████▌                               | 1374/4580 [18:38<33:27,  1.60img/s, loss (batch)=0.0184]INFO: Validation Dice score: 0.9778353571891785
Epoch 2/5:  40%|█████████████████▌                          | 1832/4580 [25:27<29:00,  1.58img/s, loss (batch)=0.00939]INFO: Validation Dice score: 0.9890643954277039
Epoch 2/5:  50%|██████████████████████                      | 2290/4580 [32:17<23:57,  1.59img/s, loss (batch)=0.00695]INFO: Validation Dice score: 0.985427737236023
Epoch 2/5:  60%|██████████████████████████▍                 | 2748/4580 [39:06<19:11,  1.59img/s, loss (batch)=0.00713]INFO: Validation Dice score: 0.9916283488273621
Epoch 2/5:  70%|███████████████████████████████▍             | 3206/4580 [45:56<14:20,  1.60img/s, loss (batch)=0.0117]INFO: Validation Dice score: 0.989119291305542
Epoch 2/5:  80%|████████████████████████████████████▊         | 3664/4580 [52:46<09:34,  1.59img/s, loss (batch)=0.019]INFO: Validation Dice score: 0.9562455415725708
Epoch 2/5:  90%|███████████████████████████████████████▌    | 4122/4580 [59:37<04:49,  1.58img/s, loss (batch)=0.00789]INFO: Validation Dice score: 0.9743334054946899
Epoch 2/5: 100%|███████████████████████████████████████████| 4580/4580 [1:06:27<00:00,  1.59img/s, loss (batch)=0.0176]INFO: Validation Dice score: 0.9908595681190491
Epoch 2/5: 100%|███████████████████████████████████████████| 4580/4580 [1:08:30<00:00,  1.11img/s, loss (batch)=0.0176]
INFO: Checkpoint 2 saved!
Epoch 3/5:  10%|████▌                                         | 458/4580 [05:01<43:21,  1.58img/s, loss (batch)=0.0149]INFO: Validation Dice score: 0.9890327453613281
Epoch 3/5:  20%|█████████                                    | 916/4580 [11:52<38:35,  1.58img/s, loss (batch)=0.00475]INFO: Validation Dice score: 0.9915817975997925
Epoch 3/5:  30%|█████████████▏                              | 1374/4580 [18:43<33:31,  1.59img/s, loss (batch)=0.00624]INFO: Validation Dice score: 0.9891043305397034
Epoch 3/5:  40%|█████████████████▌                          | 1832/4580 [25:33<28:48,  1.59img/s, loss (batch)=0.00693]INFO: Validation Dice score: 0.9901424050331116
Epoch 3/5:  50%|██████████████████████                      | 2290/4580 [32:24<24:00,  1.59img/s, loss (batch)=0.00644]INFO: Validation Dice score: 0.9907054901123047
Epoch 3/5:  60%|██████████████████████████▍                 | 2748/4580 [39:17<19:21,  1.58img/s, loss (batch)=0.00841]INFO: Validation Dice score: 0.9883075952529907
Epoch 3/5:  70%|██████████████████████████████▊             | 3206/4580 [46:10<14:29,  1.58img/s, loss (batch)=0.00888]INFO: Validation Dice score: 0.9883167743682861
Epoch 3/5:  80%|███████████████████████████████████▏        | 3664/4580 [53:03<09:39,  1.58img/s, loss (batch)=0.00648]INFO: Validation Dice score: 0.9893230199813843
Epoch 3/5:  90%|███████████████████████████████████████▌    | 4122/4580 [59:57<04:53,  1.56img/s, loss (batch)=0.00414]INFO: Validation Dice score: 0.9920490980148315
Epoch 3/5: 100%|██████████████████████████████████████████| 4580/4580 [1:06:50<00:00,  1.58img/s, loss (batch)=0.00899]INFO: Validation Dice score: 0.9898479580879211
Epoch 3/5: 100%|██████████████████████████████████████████| 4580/4580 [1:08:52<00:00,  1.11img/s, loss (batch)=0.00899]
INFO: Checkpoint 3 saved!
Epoch 4/5:  10%|████▌                                        | 458/4580 [05:01<43:22,  1.58img/s, loss (batch)=0.00894]INFO: Validation Dice score: 0.9895857572555542
Epoch 4/5:  20%|█████████▏                                    | 916/4580 [11:52<38:18,  1.59img/s, loss (batch)=0.0196]INFO: Validation Dice score: 0.9874567985534668
Epoch 4/5:  30%|█████████████▏                              | 1374/4580 [18:43<33:49,  1.58img/s, loss (batch)=0.00499]INFO: Validation Dice score: 0.9902823567390442
Epoch 4/5:  40%|█████████████████▌                          | 1832/4580 [25:34<28:55,  1.58img/s, loss (batch)=0.00586]INFO: Validation Dice score: 0.9905810952186584
Epoch 4/5:  50%|██████████████████████                      | 2290/4580 [32:25<23:59,  1.59img/s, loss (batch)=0.00433]INFO: Validation Dice score: 0.9883981347084045
Epoch 4/5:  60%|██████████████████████████▍                 | 2748/4580 [39:16<19:15,  1.59img/s, loss (batch)=0.00793]INFO: Validation Dice score: 0.9901265501976013
Epoch 4/5:  70%|██████████████████████████████▊             | 3206/4580 [46:07<14:22,  1.59img/s, loss (batch)=0.00493]INFO: Validation Dice score: 0.9903706312179565
Epoch 4/5:  80%|███████████████████████████████████▏        | 3664/4580 [52:58<09:36,  1.59img/s, loss (batch)=0.00924]INFO: Validation Dice score: 0.9880368113517761
Epoch 4/5:  90%|███████████████████████████████████████▌    | 4122/4580 [59:50<04:49,  1.58img/s, loss (batch)=0.00935]INFO: Validation Dice score: 0.9888526201248169
Epoch 4/5: 100%|██████████████████████████████████████████| 4580/4580 [1:06:48<00:00,  1.57img/s, loss (batch)=0.00763]INFO: Validation Dice score: 0.9879897236824036
Epoch 4/5: 100%|██████████████████████████████████████████| 4580/4580 [1:08:53<00:00,  1.11img/s, loss (batch)=0.00763]
INFO: Checkpoint 4 saved!
Epoch 5/5:  10%|████▌                                        | 458/4580 [05:05<43:30,  1.58img/s, loss (batch)=0.00895]INFO: Validation Dice score: 0.9901787638664246
Epoch 5/5:  20%|█████████                                    | 916/4580 [12:01<39:06,  1.56img/s, loss (batch)=0.00613]INFO: Validation Dice score: 0.9879647493362427
Epoch 5/5:  30%|█████████████▏                              | 1374/4580 [18:58<34:07,  1.57img/s, loss (batch)=0.00776]INFO: Validation Dice score: 0.9896172285079956
Epoch 5/5:  40%|█████████████████▌                          | 1832/4580 [25:57<29:07,  1.57img/s, loss (batch)=0.00596]INFO: Validation Dice score: 0.987429141998291
Epoch 5/5:  50%|██████████████████████                      | 2290/4580 [32:58<25:58,  1.47img/s, loss (batch)=0.00955]INFO: Validation Dice score: 0.9898453950881958
Epoch 5/5:  60%|███████████████████████████                  | 2748/4580 [40:00<19:19,  1.58img/s, loss (batch)=0.0101]INFO: Validation Dice score: 0.990775465965271
Epoch 5/5:  70%|██████████████████████████████▊             | 3206/4580 [47:01<14:39,  1.56img/s, loss (batch)=0.00958]INFO: Validation Dice score: 0.9894661903381348
Epoch 5/5:  80%|███████████████████████████████████▏        | 3664/4580 [54:03<09:46,  1.56img/s, loss (batch)=0.00558]INFO: Validation Dice score: 0.9878119230270386
Epoch 5/5:  90%|█████████████████████████████████████▊    | 4122/4580 [1:01:07<05:30,  1.38img/s, loss (batch)=0.00592]INFO: Validation Dice score: 0.9925376772880554
Epoch 5/5: 100%|██████████████████████████████████████████| 4580/4580 [1:08:19<00:00,  1.58img/s, loss (batch)=0.00505]INFO: Validation Dice score: 0.9901823401451111
Epoch 5/5: 100%|██████████████████████████████████████████| 4580/4580 [1:10:24<00:00,  1.08img/s, loss (batch)=0.00505]
INFO: Checkpoint 5 saved!
wandb: Waiting for W&B process to finish... (success).
wandb: - 17.470 MB of 17.480 MB uploaded (0.000 MB deduped)
wandb: Run history:
wandb:           epoch ▁▁▁▁▁▁▁▁▃▃▃▃▃▃▃▃▅▅▅▅▅▅▅▅▆▆▆▆▆▆▆▆████████
wandb:   learning rate ███████████████▂▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb:            step ▁▁▁▁▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▆▇▇▇▇▇███
wandb:      train loss █▅▂▂▂▂▂▁▁▁▂▁▂▁▁▁▁▁▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb: validation Dice ▁▅▄▅▅▃▆▇▇▇▆███▃▅████▇▇███▇████▇██▇█▇██▇█
wandb:
wandb: Run summary:
wandb:           epoch 5
wandb:   learning rate 0.0
wandb:            step 22900
wandb:      train loss 0.00505
wandb: validation Dice 0.99018
wandb:
wandb: Synced rare-pyramid-2: https://wandb.ai/anony-moose-446188/U-Net/runs/37zf8lx8?apiKey=9376384096a26debc2c33745dfffa759f3ac9348
wandb: Synced 6 W&B file(s), 150 media file(s), 0 artifact file(s) and 0 other file(s)
wandb: Find logs at: .\wandb\run-20221122_093746-37zf8lx8\logs
```

## Predict
```
$ python predict.py --model checkpoints/checkpoint_epoch5.pth --input 29bb3ece3180_11.jpg
---
(Nothing printed)
```
Nothing printed, but it works!!

![Predict Image](/29bb3ece3180_11_OUT.png)

<details><summary>View original README.md</summary><div>
# U-Net: Semantic segmentation with PyTorch
<a href="#"><img src="https://img.shields.io/github/workflow/status/milesial/PyTorch-UNet/Publish%20Docker%20image?logo=github&style=for-the-badge" /></a>
<a href="https://hub.docker.com/r/milesial/unet"><img src="https://img.shields.io/badge/docker%20image-available-blue?logo=Docker&style=for-the-badge" /></a>
<a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-v1.9.0-red.svg?logo=PyTorch&style=for-the-badge" /></a>
<a href="#"><img src="https://img.shields.io/badge/python-v3.6+-blue.svg?logo=python&style=for-the-badge" /></a>

![input and output for a random image in the test dataset](https://i.imgur.com/GD8FcB7.png)


Customized implementation of the [U-Net](https://arxiv.org/abs/1505.04597) in PyTorch for Kaggle's [Carvana Image Masking Challenge](https://www.kaggle.com/c/carvana-image-masking-challenge) from high definition images.

- [Usage Example by Ryo(me)](#usage-example-by-ryome)
  - [background](#background)
  - [Installation](#installation)
    - [Problem #1](#problem-1)
    - [How to solve Problem #1](#how-to-solve-problem-1)
  - [Train](#train)
  - [Predict](#predict)
- [U-Net: Semantic segmentation with PyTorch](#u-net-semantic-segmentation-with-pytorch)
  - [Quick start](#quick-start)
    - [Without Docker](#without-docker)
    - [With Docker](#with-docker)
  - [Description](#description)
  - [Usage](#usage)
    - [Docker](#docker)
    - [Training](#training)
    - [Prediction](#prediction)
  - [Weights \& Biases](#weights--biases)
  - [Pretrained model](#pretrained-model)
  - [Data](#data)

## Quick start

### Without Docker

1. [Install CUDA](https://developer.nvidia.com/cuda-downloads)

2. [Install PyTorch](https://pytorch.org/get-started/locally/)

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Download the data and run training:
```bash
bash scripts/download_data.sh
python train.py --amp
```

### With Docker

1. [Install Docker 19.03 or later:](https://docs.docker.com/get-docker/)
```bash
curl https://get.docker.com | sh && sudo systemctl --now enable docker
```
2. [Install the NVIDIA container toolkit:](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
   && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
   && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```
3. [Download and run the image:](https://hub.docker.com/repository/docker/milesial/unet)
```bash
sudo docker run --rm --shm-size=8g --ulimit memlock=-1 --gpus all -it milesial/unet
```

4. Download the data and run training:
```bash
bash scripts/download_data.sh
python train.py --amp
```

## Description
This model was trained from scratch with 5k images and scored a [Dice coefficient](https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient) of 0.988423 on over 100k test images.

It can be easily used for multiclass segmentation, portrait segmentation, medical segmentation, ...


## Usage
**Note : Use Python 3.6 or newer**

### Docker

A docker image containing the code and the dependencies is available on [DockerHub](https://hub.docker.com/repository/docker/milesial/unet).
You can download and jump in the container with ([docker >=19.03](https://docs.docker.com/get-docker/)):

```console
docker run -it --rm --shm-size=8g --ulimit memlock=-1 --gpus all milesial/unet
```


### Training

```console
> python train.py -h
usage: train.py [-h] [--epochs E] [--batch-size B] [--learning-rate LR]
                [--load LOAD] [--scale SCALE] [--validation VAL] [--amp]

Train the UNet on images and target masks

optional arguments:
  -h, --help            show this help message and exit
  --epochs E, -e E      Number of epochs
  --batch-size B, -b B  Batch size
  --learning-rate LR, -l LR
                        Learning rate
  --load LOAD, -f LOAD  Load model from a .pth file
  --scale SCALE, -s SCALE
                        Downscaling factor of the images
  --validation VAL, -v VAL
                        Percent of the data that is used as validation (0-100)
  --amp                 Use mixed precision
```

By default, the `scale` is 0.5, so if you wish to obtain better results (but use more memory), set it to 1.

Automatic mixed precision is also available with the `--amp` flag. [Mixed precision](https://arxiv.org/abs/1710.03740) allows the model to use less memory and to be faster on recent GPUs by using FP16 arithmetic. Enabling AMP is recommended.


### Prediction

After training your model and saving it to `MODEL.pth`, you can easily test the output masks on your images via the CLI.

To predict a single image and save it:

`python predict.py -i image.jpg -o output.jpg`

To predict a multiple images and show them without saving them:

`python predict.py -i image1.jpg image2.jpg --viz --no-save`

```console
> python predict.py -h
usage: predict.py [-h] [--model FILE] --input INPUT [INPUT ...] 
                  [--output INPUT [INPUT ...]] [--viz] [--no-save]
                  [--mask-threshold MASK_THRESHOLD] [--scale SCALE]

Predict masks from input images

optional arguments:
  -h, --help            show this help message and exit
  --model FILE, -m FILE
                        Specify the file in which the model is stored
  --input INPUT [INPUT ...], -i INPUT [INPUT ...]
                        Filenames of input images
  --output INPUT [INPUT ...], -o INPUT [INPUT ...]
                        Filenames of output images
  --viz, -v             Visualize the images as they are processed
  --no-save, -n         Do not save the output masks
  --mask-threshold MASK_THRESHOLD, -t MASK_THRESHOLD
                        Minimum probability value to consider a mask pixel white
  --scale SCALE, -s SCALE
                        Scale factor for the input images
```
You can specify which model file to use with `--model MODEL.pth`.

## Weights & Biases

The training progress can be visualized in real-time using [Weights & Biases](https://wandb.ai/).  Loss curves, validation curves, weights and gradient histograms, as well as predicted masks are logged to the platform.

When launching a training, a link will be printed in the console. Click on it to go to your dashboard. If you have an existing W&B account, you can link it
 by setting the `WANDB_API_KEY` environment variable. If not, it will create an anonymous run which is automatically deleted after 7 days.


## Pretrained model
A [pretrained model](https://github.com/milesial/Pytorch-UNet/releases/tag/v3.0) is available for the Carvana dataset. It can also be loaded from torch.hub:

```python
net = torch.hub.load('milesial/Pytorch-UNet', 'unet_carvana', pretrained=True, scale=0.5)
```
Available scales are 0.5 and 1.0.

## Data
The Carvana data is available on the [Kaggle website](https://www.kaggle.com/c/carvana-image-masking-challenge/data).

You can also download it using the helper script:

```
bash scripts/download_data.sh
```

The input images and target masks should be in the `data/imgs` and `data/masks` folders respectively (note that the `imgs` and `masks` folder should not contain any sub-folder or any other files, due to the greedy data-loader). For Carvana, images are RGB and masks are black and white.

You can use your own dataset as long as you make sure it is loaded properly in `utils/data_loading.py`.


---

Original paper by Olaf Ronneberger, Philipp Fischer, Thomas Brox:

[U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)

![network architecture](https://i.imgur.com/jeDVpqF.png)
</div></details>