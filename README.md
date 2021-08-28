# autoencoder_program_synthesis

Autoencoder Program Synthesis: A Tree-based VAE-RNN as a tool for generating new C++ programs.

## Requirements
1. Make sure to have PyTorch installed: https://pytorch.org/get-started/locally/.
2. Make sure to install clang LLVM https://github.com/llvm/llvm-project/releases/tag/llvmorg-12.0.1, for windows make sure to add LLVM to the system path
3. Install clang-format for automatic formatting of the generated code, run: ```apt-get install clang-format```
4. To install this as a package run: ```pip install git+https://github.com/sander102907/autoencoder_program_synthesis```


## Set up
1. Download a pretrained model checkpoint from [here](https://surfdrive.surf.nl/files/index.php/s/4L8v2RaPtEqCxTg/download)
2. Unzip the contents and place the checkpoint in a folder
3. Locate the libclang.so library file on your computer and save the path

## Usage
Then you can simply import the package in your python code and encode C++ programs to latent vectors and decode latent vectors back to strings. The encoding function takes a program represented as a string and the decoder takes a torch tensor, and some sampling parameters: temperature, top_k and top_p (see this [paper](https://arxiv.org/abs/1904.09751) for more information). Use temperature=0, top_k=0 and top_p=0 for greedy decoding. The pretrained model that is provided in the set up works with latent vectors of size 150, hence the decoder takes a torch tensor of shape [X, 150] where X can be any number of programs you want to generate. An example of a reconstruction:

```
from autoencoder_program_synthesis.t2t_vae import Tree2Tree

libclang_path = '/usr/lib/x86_64-linux-gnu/libclang-6.0.so.1'
checkpoint_path = 'checkpoints/1/'

tree2tree = Tree2Tree(libclang_path, checkpoint_path)

program = """
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(NULL);

    int n;
    cin >> n;
    map<int,int> cn;
    int ans = 0;

    for(int i = 0; i < n; i++) {
        int x;
        cin >> x;
        cn[x]++;
        ans = max(ans,cn[x]);
    }

    cout << ans;
    return 0;
}"""

z, decl_names = tree2tree.encode(program)
reconstructed_program = tree2tree.decode(z, temperature=0.7, top_k=40, top_p=0.9, declared_names=decl_names)
```

## TODO

Model training/testing/finetuning
