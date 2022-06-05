# TopFormer Cloth Segmentation
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1dKLZ0qxVFUffAR6lMknvMbVbpBMi0kqU?usp=sharing)

# Part 1
## Project structure
The project has 2 folders sevaral folders:
* u2net contains utils for running the u2net from the [repo](https://github.com/levindabhi/cloth-segmentation). I use it for inference and for metric calculation.
* topformer - TopFormer code for training and inference.
* common_segmentation - folder which contains code for both previous folders. It's a small package which I install with `pip install -e`.

Generally I would use [pytorch-lightning](https://github.com/PyTorchLightning/pytorch-lightning) to make it a single codebase and use [segmentation_models_pytorch](https://github.com/qubvel/segmentation_models.pytorch) to base my code upon (configuring experiments with [Hydra](https://github.com/facebookresearch/hydra)), but due to the time limits I decided to split the experiments into several folders.  I would also use a more advanced and comfortable experiment tracking service, such as WanDB or Neptune instead of tensorboard.

I just felt like doing it in a very quck hacky way, as this was not intended to be a large codebase.

## Data
As the U2NET model my TopFormer is also trained on the [IMaterialist Dataset](https://www.kaggle.com/c/imaterialist-fashion-2019-FGVC6/data). The comparison is a flawed: I'm using 15% of the dataset as validation and later test my best weights on the [People Clothing Dataset](https://www.kaggle.com/datasets/rajkumarl/people-clothing-segmentation). As far as I can understand the U2NET model was trained on the whole IMaterialist Dataset for 100k steps. I did not know any magic numbers so I had to rely on a validation set.

I also train TopFormer on 768x768 rescaled images for better comparison with U2NET.

As a side note: I believe there may be inconsistent/bad markup for the train and the test dataset. The labels might not be parsed in a correct way, some code arouses suspicion. Nonetheless I tried to make the training conditions as fair as possible.


## Experiments
Results on People Clothing Dataset.

I've made several experiments:
| MODEL                            | Upper | Lower | Whole body |
|----------------------------------|-------|-------|------------|
| U2NET                            | **41.8**  | 37.7  | 28.5       |
| TopFormer CE                     | 40.4  | 38.1  | 28.4       |
| TopFormer CE + Dice              | 41.6  | 37.0  | **29.1**       |
| TopFormer CE + Dice + Heavy Augs | 41.2  | **47.3**  | 21.7       |

TopFormer CE was made with config topformer/config.py
Other models were trained with topformer/conf_20.py and initialized from the weights 
from [original repo](https://github.com/hustvl/TopFormer) with the model TopFormer-B_512x512_4x8_160k (trained on ade20k).
The weighs can be accessed on [Google Drive](https://drive.google.com/drive/folders/16j22QitHAiX4Sf2Ap9eSZAtnFezl5ROb?usp=sharing).

I also tried finetuning the model with Lovasz loss after training it with CrossEntropy and Dice losses but it did not improve the results. This would probably be beneficial, but I did not have enough time to run enough experiments for a solid conclusion.

**Memory**:

* U2NET model has 40M parameters, which results in a weight (FP32) of 160Mb.
* TopFormer model has 5M parameters, which results in a weight (FP32) of 20Mb.

N.B. The TopFormer checkpoint also contains state dicts of the optimizer and scheduler, which make it 58Mb.

**Time**:

On the People Clothing Dataset on Tesla V100 32GB to process 1000 images (batchsize 1):
* U2Net takes 0.045s per image
* TopFormer takes 0.022s per image

## Inference scripts
`--input-dir` is the path to unarchived People Clothing Dataset

**U2NET**

```bash
python u2net/infer_on_ccp.py --input-dir ../arch --load-path cloth_segm_u2net_latest.pth
```

**TopFormer**

```
python topformer/infer_on_ccp.py --input-dir ../arch/ --load-path checkpoint/model.pt [--torch-transforms]
```
Use `--torch-transforms` for models TopFormer CE, TopFormer CE + Dice.

Inference can also be tested in [Google Colab](https://colab.research.google.com/drive/1dKLZ0qxVFUffAR6lMknvMbVbpBMi0kqU?usp=sharing)

## Quality comparison
I gathered a few images for demonstrational purposes. Images can be accessed on [Google Drive](https://drive.google.com/drive/folders/16j22QitHAiX4Sf2Ap9eSZAtnFezl5ROb?usp=sharing).

Comparison of TopFormer and U2NET can be seen in a [nearby folder](https://drive.google.com/drive/folders/1WfJ4jKg55R-9Wy9e7u569eMHwLsOBDoT?usp=sharing).


# Part 2
Question:
> Check the “Dress Code” dataset. Based on the annotations offered by this dataset which include landmarks, segments and skeleton - do you believe training with that dataset would give better results than the one you’ve had with the TopFormer model? Explain why.

I think the Dress Code dataset would improve the model quality. I do believe that the main improvement would come from the better quality of the labeled data.

I believe that the skeleton and landmark markup could be used to train an additional head for the model. The additional loss would further regularize the model and improve the quality. Although this conclusion seems intuitive in practice additional losses do not always improve model performance, so ultimately it will come down to experimental evidence whether this helps or not.

# Bonus question
Question:
> For porting the trained model to mobile devices, what kind of methods or modification can you use to reduce the model size and accelerate the speed? List some you know.

1. Running in fp16 instead of fp32. Normalization layers should stay in fp32 to avoid overflows and numerical instability.
2. Operation fusing for data bound operations. Greatly explained in https://horace.io/brrr_intro.html. Tracing can be used with the automatic fusing compiler.
3. Training a smaller model. TopFormers scale down to 1M parameters in the original work. This comes at a sacrifice of model quality.
4. There are other ways of improving model quality without changing it: finetuning with other loss after training at a smaller learning rate, using [Hard Example Mining](https://arxiv.org/abs/1604.03540v1)
5. Use quantization to INT8. [Supported in TNN for mobile devices](https://github.com/Tencent/TNN)
6. Using tracing to compile model to a static graph https://pytorch.org/docs/stable/generated/torch.jit.trace.html.  
7. Writing specialized low-level code to preallocate memory buffers (to fight memory fragmentation and for faster transfers) and use pinned memory.
8. Pruning model weights (again at the cost of quality) https://jacobgil.github.io/deeplearning/pruning-deep-learning, https://ai.googleblog.com/2021/03/accelerating-neural-networks-on-mobile.html
9. Neural network can be distilled into a smaller network with distillation (KL over labels) https://arxiv.org/pdf/1503.02531.pdf. This may result in a smaller network with a comparable quality, but then again: it does not guarantee great results and multiple experiments are required.

Most of the techniques are implemented in frameworks for mobile inference such as [TNN](https://github.com/Tencent/TNN) and [ncnn](https://github.com/Tencent/ncnn).