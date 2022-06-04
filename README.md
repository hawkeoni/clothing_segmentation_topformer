# TopFormer Cloth Segmentation


## Project structure
The project has 2 folders sevaral folders:
* u2net contains utils for running the u2net from the [repo](https://github.com/levindabhi/cloth-segmentation). I use it for inference and for metric calculation.
* topformer - TopFormer code for training and inference.
* common_segmentation - folder which contains code for both previous folders. It's a small package which I install with `pip install -e`.

Generally I would use [pytorch-lightning](https://github.com/PyTorchLightning/pytorch-lightning) to make it a single codebase and use [segmentation_models_pytorch](https://github.com/qubvel/segmentation_models.pytorch) to base my code upon, but due to the time limits I decided to split the experiments into several folders.  I would also use a more advanced and comfortable experiment tracking service, such as WanDB or Neptune.

I just felt like doing it in a very quck way, as this was not intended to be a large codebase.


## Data
As the U2NET model my TopFormer is also trained on the [IMaterialist Dataset](https://www.kaggle.com/c/imaterialist-fashion-2019-FGVC6/data). The comparison is a flawed: I'm using 15% of the dataset as validation and later test my best weights on the [People Clothing Dataset][https://www.kaggle.com/datasets/rajkumarl/people-clothing-segmentation]. As far as I can understand the U2NET model was trained on the whole IMaterialist Dataset.

I also train TopFormer on 768x768 rescaled images for better comparison with U2NET.


TODO: Which architecture I use, speed, quality
