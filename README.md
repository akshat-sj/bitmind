# BitMind

## Description
Introducing BitMind, your AI-powered gaming voice assistant. Tired of pausing your game to search for information? Spending hours browsing the web to get some information that shouldn't be hard to get?  Miss those guide books that never needed an internet connection to work ? BitMind is here to revolutionize your gaming experience. With just a command, get instant answers to your questions, real-time strategies, and personalized tips tailored to your playstyle.


## Table of Contents

- [BitMind](#bitmind)
  - [Description](#description)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Acknowledgements](#acknowledgements)

## Installation

To install and run this project, follow these steps:

1. Clone the repository.
   ```bash
   ```
2. Create conda environment based on provided yaml file
   ```bash
   conda env create --file=env.yaml
   conda activate ryzenai-transformers
   ```
3. Setup environment
   ```bash
   .\setup.bat
   ```
4. Build dependancies 
   ```bash
   pip install ops\cpp --force-reinstall
   ```


## Usage

To use this project, follow these steps:

1. We need to setup the environment every time the conda environment is disconnected 
   ```bash
   .\setup.bat
   ```
2. Download weights and quantize the model 
   ```bash
   python run_awq.py --w_bit 4 --task quantize 
   ```
3. Running the makefile now
   ```bash
   cd llama2
   make
   ```
## Acknowledgements
I would like to express my sincere gratitude to AMD for providing the mini PC and software, enabling me to participate in the hackathon.

