# Programming Assignment 4
Welcome to CS224N Project Assignment 4 Reading Comprehension.
The project has several dependencies that have to be satisfied before running the code. You can install them using your preferred method -- we list here the names of the packages using `pip`.

# Requirements

The starter code provided pressuposes a working installation of Python 2.7, as well as a TensorFlow 0.12.1.

It should also install all needed dependnecies through
`pip install -r requirements.txt`.

# Running your assignment

You can get started by downloading the datasets and doing dome basic preprocessing:

$ code/get_started.sh

Note that you will always want to run your code from this assignment directory, not the code directory, like so:

$ python code/train.py

This ensures that any files created in the process don't pollute the code directoy.


# Description

A Dynamic Coattention Network.

Train using python code/train.py with the flags:

--learning_rate (0.01) : initial ADAM learning rate (0.001 seems to work well with batch 32)
--batch_size (32)
--train_dir (train) : the directory the checkpointer will write save files to (watch out for overwriting!)
--load_train_dir (train) : the directory to load model from (defaults to train_dir but probably safer to keep different)
--log_dir (log) : directory where the log and tensorboard files will go
--optimizer (adam) : adam or sgd




