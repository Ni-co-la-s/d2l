# D2l

Personal project going through the "Dive into Deep Learning" book and coding things I find interesting.

## Description

Implementing chapters I find interesting from the book.
For learning purposes, almost all the code is written manually, without LSP.
To test the implementations, I add training experiments that can be ran with wandb.

Currently done:
- Chapter 07
- Chapter 08

## Chapter 07 : Convolutional Neural Networks

Chapter 07 covers the intuition and maths behind Convolutional Layers and Convolutional Neural networks.

### Modules implementation

7.01 through 7.05 covers all the necessary modules to build the basic LeNet.
They are implemented from scratch in this [modules.py file](ch_07_Convolutional_Neural_Network/modules.py):

#### Convolutional layers

Implemented two versions:
- Conv2dNotOpti: Slow version that uses 7 nested loops to compute the output.
- Conv2d: Faster version that uses im2col to reduce the complexity.

Tested them by comparing the outputs of their forward pass to [nn.Conv2d](https://docs.pytorch.org/docs/2.11/generated/torch.nn.Conv2d.html) in [test_ch07.py::TestModules](tests/test_ch07.py).

This does not support all parameters used by the torch implemention (no groups, dilation...)

#### Pool layers

Implemented MaxPool2d.
Deviates from the book which used AvgPool2d to respect the LeNet design.

Tested them by comparing the outputs of its forward pass to [nn.MaxPool2d](https://docs.pytorch.org/docs/2.11/generated/torch.nn.MaxPool2d.html) in [test_ch07.py::TestModules](tests/test_ch07.py).

This does not support all parameters used by the torch implemention (dilation, return_indices...)

#### Linear layers

Implemented Linear.
This is not from this chapter, but ended up doing it so that I can test with all manual layers.

Tested them by comparing the outputs of its forward pass to [nn.Linear](https://docs.pytorch.org/docs/2.11/generated/torch.nn.modules.linear.Linear.html) in [test_ch07.py::TestModules](tests/test_ch07.py).


### CNN implementation

7.06 covers the design of the original LeNet model.
My implementation uses MaxPool instead of AvgPool.
I avoid Lazy layers so that I have to compute the shapes at every step. 
It is implemented in this [cnn.py file](ch_07_Convolutional_Neural_Network/cnn.py):

#### Model implementation

Implemented CNN that supports 3 implementation modes:
- Implementation.TORCH: uses torch layers for convolution, max pooling and linear.
- Implementation.MANUAL_OPTI: uses implemented max pooling and linear, along with optimized implemented version for convolution.
- Implementation.MANUAL_BASE: uses implemented max pooling and linear, along with non-optimized implemented version for convolution
As mentioned before, I used MaxPool instead of the original AvgPool

#### Training

Training uses a config that supports the following:
- implem: Implementation.TORCH, Implementation.MANUAL_OPTI, Implementation.MANUAL_BASE
- device: "cuda" or "cpu"
- num_epochs: int
- batch_size: int
- optim: Optim.SGD or Optim.ADAM
- lr: float
- project_name: name of your wandb project or None if you don't want wandb
- run_name: name of your wandb run name or None if you want it to be automatically generated
- job_type: name of your wandb job type or None
- dataset: DatasetVersion.MNIST or DatasetVersion.FASHION_MNIST, based on which dataset you want to train on

To ensure results are the same everytime, a seed is set at the beginning

#### Wandb experiments

Add options for training in wandb. It logs the following:
- epoch
- train step metrics (loss and accuracy)
- train epochs metrics (loss and accuracy)
- validation epochs metrics (loss and accuracy)
- Images for the activations of the model for convolutions and linear layers. It does it for one real input and one noise input.

#### Results

Results deviate from the ones visible in d2l.ai.
The main reasons is the use of ReLU + MaxPool instead of Sigmoid + AvgPool.
You can find below the graph of training and val loss and accuracy for FASHION MNIST (Used in the book) and MNIST.
As well as the activations throughout the final model, for an actual input and a noisy input (Those were questions at the end of the chapter).


##### Fashion-MNIST

<p align="center">
  <img src="images/ch07_fashion_mnist_metrics.png" height="280"/>
</p>

<p align="center">
  <img src="images/ch07_fashion_mnist_activation_real_input.png" height="220"/>
  <img src="images/ch07_fashion_mnist_activation_noisy_input.png" height="220"/><br/>
  <em>Left: Real input — Right: Noisy input</em>
</p>

---

##### MNIST

<p align="center">
  <img src="images/ch07_mnist_metrics.png" height="280"/>
</p>

<p align="center">
  <img src="images/ch07_mnist_activation_real_input.png" height="220"/>
  <img src="images/ch07_mnist_activations_noisy_input.png" height="220"/><br/>
  <em>Left: Real input — Right: Noisy input</em>
</p>
