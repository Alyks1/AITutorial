# AITutorial

This Deep Learning algorithm was written using the [Neural Network Book by Michael Nielsen](http://neuralnetworksanddeeplearning.com/chap1.html).

## How to start it
1. Clone this repository
2. Install Python
3. Create a virtual environment `$ python -m venv env`
4. Source the virtual environment:
    - Mac OS/Linux: `$ source env/bin/activate`
    - Windows: `$ source env/Scripts/activate`
5. Install the requirements.txt `$ pip install -r requirements.txt`
6. Start the program by running `$ python src/network.py`
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

```Epoch 0: 7463 / 10000
Epoch 1: 7547 / 10000
Epoch 2: 7647 / 10000
Epoch 3: 7652 / 10000
Epoch 4: 7676 / 10000
Epoch 5: 7698 / 10000
Epoch 6: 7691 / 10000
Epoch 7: 7703 / 10000
Epoch 8: 7714 / 10000
Epoch 9: 7731 / 10000
Epoch 10: 7720 / 10000
Epoch 11: 7743 / 10000
Epoch 12: 7724 / 10000
Epoch 13: 7744 / 10000
Epoch 14: 7752 / 10000
Epoch 15: 7761 / 10000
Epoch 16: 7755 / 10000
Epoch 17: 7749 / 10000
Epoch 18: 7793 / 10000
Epoch 19: 7866 / 10000
Epoch 20: 8287 / 10000
Epoch 21: 8674 / 10000
Epoch 22: 8686 / 10000
Epoch 23: 8682 / 10000
Epoch 24: 9524 / 10000
Epoch 25: 9586 / 10000
Epoch 26: 9572 / 10000
Epoch 27: 9583 / 10000
Epoch 28: 9568 / 10000
Epoch 29: 9586 / 10000
```
