# GhostSANet
[BMVC 2021] GhostShiftAddNet: More Features from Energy-Efficient Operations
It partly uses two projects including ShiftAddNet and GhostNet: 
***ShiftAddNet: A Hardware-Inspired Deep Network*** published on the NeurIPS 2020 and  ***GhostNet: More Features from Cheap Operations*** published on the CVPR 2020.

---

## Prerequisite

* GCC >= 5.4.0
* PyTorch >= 1.4
* Other common library are included in `requirements.txt`


### We use compiler of Adder Cuda Kernal from ShiftAddNet 

The original [AdderNet Repo](https://github.com/huawei-noah/AdderNet) considers using PyTorch for implementing add absed convolution, however it remains slow and requires much more runtime memory costs as compared to the variant with CUDA acceleration.

We here provide one kind of CUDA implementation, please follow the intruction below to compile and check that the `forwad/backward` results are consistent with the original version.

#### Step 1: modify PyTorch before launch (for solving compiling issue)

Change lines:57-64 in `anaconda3/lib/python3.7/site-packages/torch/include/THC/THCTensor.hpp`
from:
````
#include <THC/generic/THCTensor.hpp>
#include <THC/THCGenerateAllTypes.h>

#include <THC/generic/THCTensor.hpp>
#include <THC/THCGenerateBoolType.h>

#include <THC/generic/THCTensor.hpp>
#include <THC/THCGenerateBFloat16Type.h>
````
to:
````
#include <THC/generic/THCTensor.h>
#include <THC/THCGenerateAllTypes.h>

#include <THC/generic/THCTensor.h>
#include <THC/THCGenerateBoolType.h>

#include <THC/generic/THCTensor.h>
#include <THC/THCGenerateBFloat16Type.h>
````

#### Step 2: launch command to make sure you can successfully compile


````
python check.py
````

You should be able to successfully compile and see the runtime speed comparisons in the toy cases.



## Reproduce Results in Paper

We release the pretrained checkpoints in [Google Drive](https://drive.google.com/drive/folders/1nON7w5-y40PPGT1NCh_n_h3RLFwP8DO6?usp=sharing). To evaluate the inference accuracy of test set, we provide evaluation scripts shown below for your convenience. If you want to train your own model, the only change should be removing `--eval_only` option in the commands.

* Examples for training of AdderNet

````
# CIFAR-10
    bash ./scripts/addernet/cifar10/FP32.sh
    bash ./scripts/addernet/cifar10/FIX8.sh

# CIFAR-100
    bash ./scripts/addernet/cifar100/FP32.sh
    bash ./scripts/addernet/cifar100/FIX8.sh
````

* Examples for training of DeepShift

````
# CIFAR-10
    bash ./scripts/deepshift/cifar10.sh

# CIFAR-100
    bash ./scripts/deepshift/cifar100.sh
````

* Examples for training of GhostShiftAddNet

````
# CIFAR-10/CIFAR-100/ImageNet
    bash ./scripts/shiftaddnet/cifar10/ghostFP32.sh
    bash ./scripts/shiftaddnet/cifar10/ghostFIX8.sh

````
## Main results of GhostShiftAddNet on GPU/CPU and Jetson Nano
![Screenshot 2021-08-22 at 12-13-26 British_Machine_Vision_Conference pdf](https://user-images.githubusercontent.com/9842386/130353017-6ec2ce27-d16b-448a-b8ab-eb51418d2bec.png)
![Screenshot 2021-08-22 at 12-16-01 British_Machine_Vision_Conference pdf](https://user-images.githubusercontent.com/9842386/130353063-bdd35d4f-a8ee-4822-8587-57cae0495ebb.png)
