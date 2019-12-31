import numpy as np


def ask_for_feature(feature_index) -> float:
    ok = False
    val_str = 0.0
    while not ok:
        val_str = input("Enter 0 or 1 for column " + str(feature_index) + ": ")
        ok = val_str == '0' or val_str == '1'
    val = int(val_str)
    return val


def mode() -> int:
    mode_ok = False
    mode_input = 0
    while not mode_ok:
        mode_input = input("Enter:\n"
                           "(1) - Inference by rules\n"
                           "(2) - Inference by NN\n"
                           "(3) - Inference on both\n"
                           "(4) - Inference in dialog\n"
                           "(5) - Compare both approaches\n")

        mode_ok = mode_input == '1' or mode_input == '2' or mode_input == '3' or mode_input == '4' or mode_input == '5'
    return int(mode_input)


def data_line() -> np.ndarray:

    data_ok = False
    data_str = ""
    while not data_ok:
        data_raw = input("Enter one line of data: ")
        data_str = np.array(data_raw.split(','))
        if not len(data_str) == 16:
            continue
        data_ok = True
        for val in data_str:
            if not val == '0.0' and not val == '1.0':
                print('Incorrect input!')
                data_ok = False
                break
    arr = np.array(data_str.astype(np.float))
    return np.array(arr)
