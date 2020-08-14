# DSM (Deep Specification Miner) [[pdf](https://github.com/lebuitienduy/DSM/blob/master/paper/DSM.pdf)] [[slides](https://github.com/lebuitienduy/DSM/blob/master/paper/DSM-ISSTA.pptx)]
## Installation
### Linux

- Download and install Anaconda3 4.2 from https://repo.continuum.io/archive/Anaconda3-4.2.0-Linux-x86_64.sh
- After installation, include path of "bin" folder of the above Anaconda to PATH variable
- Install Tensorflow 0.12 for the installed Anaconda version using the command: 
```
conda install -c jjhelmus tensorflow=0.12.0
```
- Install graphviz for Python:
```
python -m install graphviz
```
- Test installation:
  ```
  cd data/StringTokenizer
  bash execute.sh
```
