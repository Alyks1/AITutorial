# AITutorial

This Deep Learning algorithm was written using the [Neural Network Book by Michael Nielsen](http://neuralnetworksanddeeplearning.com/chap1.html).

## How to start it
1. Clone this repository
2. Install Python
3. Create a virtual environment `$ python -m venv /path/to/new/virtual/environment`
4. Source the virtual environment:
    - Mac OS/Linux: `$ source venv/bin/activate`
    - Windows: `$ source venv/Scripts/activate`
5. Install the requirements.txt `$ pip install -r requirements.txt`
6. Start the program by starting `network.py`. For example `$ python src/network.py`
7. Wait

## Output
The output of the program displays the different epochs and a number corresponding to the amount of images the program has classified correctly.
For example:
```Epoch 0: 9014 / 10000
Epoch 1: 9323 / 10000
Epoch 2: 9410 / 10000
Epoch 3: 9406 / 10000
Epoch 4: 9473 / 10000
Epoch 5: 9475 / 10000
Epoch 6: 9495 / 10000
Epoch 7: 9488 / 10000
Epoch 8: 9563 / 10000
Epoch 9: 9567 / 10000
Epoch 10: 9595 / 10000
```
Here, after 10 Epochs the Network distinguishes 9595 out of 10000 images correctly. Thats almost 96%!
