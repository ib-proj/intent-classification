# Dialogue Act Classification Project

This project aims to classify dialogue acts in conversational data using various model architectures and input features. The goal is to compare the performance of these models and determine which combination of architecture and input features yields the best results.

The code in this project implements the method described in the paper available [here](https://www.overleaf.com/project/6405f87868c76164ec1a3a83).


## Installation

To install the necessary requirements, run:

``` pip install -r requirements.txt ```

## Run the model

To run the model, pass the model name to the `main.py` script in your terminal:

```python main.py --modelname=<model name>```


By default, the model uses a GRU encoder. To specify a different model, pass the model name as an argument.


