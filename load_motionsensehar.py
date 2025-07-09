import numpy as np

def load_motionsensehar(train_file='MotionSenseHAR_TRAIN.ts', test_file='MotionSenseHAR_TEST.ts'):
    """
    Load the MotionSenseHAR dataset from .ts files.

    Returns:
        X_train: numpy array, shape (num_train_samples, channels, time_steps)
        y_train: numpy array, shape (num_train_samples,)
        X_test: numpy array, shape (num_test_samples, channels, time_steps)
        y_test: numpy array, shape (num_test_samples,)
    """
    X_train, y_train = read_ts_file(train_file)
    X_test, y_test = read_ts_file(test_file)
    return X_train, y_train, X_test, y_test

def read_ts_file(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    #Skipping the metadata lines
    data_lines = [line.strip() for line in lines if not line.startswith('@') and line.strip()]

    X = []
    y = []

    for line in data_lines:
        parts = line.split(':')
        label_str = parts[-1].strip()
        label = label_str  # keep label as string

        channel_strs = parts[:-1]

        channels = []
        for ch_str in channel_strs:
            ch_str = ch_str.strip()
            values = [float(v) for v in ch_str.split(',')]
            channels.append(values)

        arr = np.array(channels)
        X.append(arr)
        y.append(label)

    return np.array(X), np.array(y)





