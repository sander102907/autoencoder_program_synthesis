# autoencoder_program_synthesis

Autoencoder Program Synthesis: A Tree-based VAE-RNN as a tool for generating new C++ programs.

## Requirements
1. Make sure to have PyTorch installed: https://pytorch.org/get-started/locally/.
2. To install this as a package run: ```pip install git+https://github.com/sander102907/autoencoder_program_synthesis```


## Set up
1. Download a pretrained model checkpoint from [here](https://surfdrive.surf.nl/files/index.php/s/4L8v2RaPtEqCxTg/download)
2. Unzip the contents and place the checkpoint in a folder
3. Locate the libclang.so library file on your computer and save the path

## Usage
Then you can simply import the package in your python code and encode C++ programs to latent vectors and decode latent vectors back to strings. The encoding function takes a program represented as a string and the decoder takes a torch tensor, and some sampling parameters: temperature, top_k and top_p (see this [paper](https://arxiv.org/abs/1904.09751) for more information). The pretrained model that is provided in the set up works with latent vectors of size 150, hence the decoder takes a torch tensor of shape [X, 150] where X can be any number of programs you want to generate. An example of a reconstruction:

```
from autoencoder_program_synthesis.t2t_vae import Tree2Tree

libclang_path = '/usr/lib/x86_64-linux-gnu/libclang-6.0.so.1'
checkpoint path = 'checkpoints/1/'

tree2tree = Tree2Tree(libclang_path, checkpoint_path)

program = """
// program to add two numbers using a function

#include <iostream>

using namespace std;

// declaring a function
int add(int a, int b) {
    return (a + b);
}"""

z = tree2tree.encode(program)
reconstructed_program = tree2tree.decode(z, temperature=0.7, top_k=40, top_p=0.9)
```

## TODO

Model training/testing/finetuning
