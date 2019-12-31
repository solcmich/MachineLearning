from NeuralNetwork import NeuralNetwork
import pandas as pd
import random
import Input
from DecisionTree import DecisionTree

DATA_PATH = 'data.csv'

REPUBLICAN = 1
DEMOCRAT = 0
INITIAL_LAYERS = [16, 1]
INITIAL_BATCH_SIZE = 6
INITIAL_EPOCHS = 10
INITIAL_LEARNING_RATE = 1.1
max_depth_tree = 5


def main():
    data = load_data()

    # DT **********************************************************************
    dt = DecisionTree(data)
    dt.ProcessRules()
    # DT **********************************************************************

    # NN ***********************************************************************
    nn = NeuralNetwork(data, INITIAL_LAYERS, INITIAL_EPOCHS, INITIAL_BATCH_SIZE, INITIAL_LEARNING_RATE)
    nn.train(10, (20, 50), (4, 30), (0, 5), (4, 60), (0.05, 0.2, 0.02))
    # NN ***********************************************************************

    while True:
        mode = Input.mode()
        if mode == 1:
            line = Input.data_line()
            infere_dt(dt, line)
        elif mode == 2:
            line = Input.data_line()
            infere_nn(nn, line)
        elif mode == 3:
            line = Input.data_line()
            infere_nn(nn, line)
            infere_dt(dt, line)
            pass
        elif mode == 4:
            # decision tree using dialog mode
            res = dt.decide_dialog()
            if res == DEMOCRAT:
                print('THIS IS DEMOCRAT BY DIALOG')
            else:
                print('THIS IS REPUBLICAN BY DIALOG')
        elif mode == 5:
            print(f'Accuracy on test data by neural network: {nn.accuracy(nn.test_data)}')
            print(f'Accuracy on test data by Decision tree: {dt.accuracy(nn.test_data)}')
            pass


def infere_nn(nn, line):
    line = line.reshape(16, 1)
    res = nn.feed_forward(line)[0][0]
    if abs(res - DEMOCRAT) < 0.5:
        print('THIS IS DEMOCRAT BY NN')
    else:
        print('THIS IS REPUBLICAN BY NN')


def infere_dt(dt, line):
    res = dt.Decide(line)
    if res == DEMOCRAT:
        print('THIS IS DEMOCRAT BY DT')
    else:
        print('THIS IS REPUBLICAN BY DT')


def load_data():
    data = pd.read_csv(DATA_PATH)
    data = data.replace(to_replace='republican', value=REPUBLICAN)
    data = data.replace(to_replace='democrat', value=DEMOCRAT)
    data_v = data.values
    random.shuffle(data_v)
    return data_v


main()
