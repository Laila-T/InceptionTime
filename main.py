from utils.constants import UNIVARIATE_ARCHIVE_NAMES as ARCHIVE_NAMES
from load_motionsensehar import load_motionsensehar
from utils.utils import transform_labels, create_directory
import numpy as np
import sys
import sklearn
import argparse
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # disables GPU



def read_all_datasets(root_dir=None, archive_name=None):
    X_train, y_train, X_test, y_test = load_motionsensehar(
        train_file='MotionSenseHAR_TRAIN.ts', test_file='MotionSenseHAR_TEST.ts')

    datasets_dict = {
        'MotionSenseHAR': {
            'train': (X_train, y_train),
            'test': (X_test, y_test)
        }
    }
    return datasets_dict


def prepare_data(datasets_dict, dataset_name):
    x_train, y_train = datasets_dict[dataset_name]['train']
    x_test, y_test = datasets_dict[dataset_name]['test']

    nb_classes = len(np.unique(np.concatenate((y_train, y_test), axis=0)))

    # Normalize labels to start at 0
    y_train, y_test = transform_labels(y_train, y_test)

    y_true = y_test.astype(np.int64)
    y_true_train = y_train.astype(np.int64)

    enc = sklearn.preprocessing.OneHotEncoder()
    enc.fit(np.concatenate((y_train, y_test), axis=0).reshape(-1, 1))

    y_train = enc.transform(y_train.reshape(-1, 1)).toarray()
    y_test = enc.transform(y_test.reshape(-1, 1)).toarray()

    if len(x_train.shape) == 2:  # univariate data
        x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
        x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

    return x_train, y_train, x_test, y_test, y_true, nb_classes, y_true_train, enc


def create_classifier(classifier_name, input_shape, nb_classes, output_directory,
                      verbose=False, build=True, clf_name=None, **kwargs):
    if classifier_name == 'nne':
        from classifiers import nne
        return nne.Classifier_NNE(output_directory, input_shape,
                                  nb_classes, verbose, clf_name=clf_name)
    if classifier_name == 'inception':
        from classifiers import inception
        return inception.Classifier_INCEPTION(output_directory, input_shape, nb_classes, verbose,
                                              build=build, **kwargs)


def fit_classifier(classifier, x_train, y_train, x_test, y_test, y_true):
    classifier.fit(x_train, y_train, x_test, y_test, y_true)


def get_xp_val(xp):
    if xp == 'batch_size':
        return [16, 32, 128]
    elif xp == 'use_bottleneck':
        return [False]
    elif xp == 'use_residual':
        return [False]
    elif xp == 'nb_filters':
        return [16, 64]
    elif xp == 'depth':
        return [3, 9]
    elif xp == 'kernel_size':
        return [8, 64]
    else:
        raise Exception('wrong argument')
    
def extract_features(classifier, x_data, output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Build feature extractor model architecture
    feature_extractor = classifier.build_feature_extractor(input_shape=x_data.shape[1:])
    
    # Load weights from the saved model (assuming best_model.keras path)
    model_weights_path = os.path.join(classifier.output_directory, 'best_model.keras')
    feature_extractor.load_weights(model_weights_path)
    
    # Predict features
    features = feature_extractor.predict(x_data, batch_size=classifier.batch_size)
    
    # Save features as numpy array
    save_path = os.path.join(output_directory, 'extracted_features.npy')
    np.save(save_path, features)
    print("Features extracted and saved to:", save_path)
    save_features_as_csv(features, output_directory)

print("Script started")
def save_features_as_csv(features, output_directory, filename='extracted_features.csv'):
    import pandas as pd
    import os
    
    # If 3D, flatten to 2D
    if len(features.shape) == 3:
        features = features.reshape(features.shape[0], -1)
    
    df = pd.DataFrame(features)
    csv_path = os.path.join(output_directory, filename)
    df.to_csv(csv_path, index=False)
    print(f"Features saved as CSV to: {csv_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['InceptionTime', 'InceptionTime_xp', 'run_length_xps'])
    parser.add_argument('--dataset', default='MotionSenseHAR')
    parser.add_argument('--train', action='store_true', help='Run training loops')
    args = parser.parse_args()
    print(f"Mode: {args.mode}, Dataset: {args.dataset}, Train flag: {args.train}")

    root_dir = 'C:\\Users\\Laila\\InceptionTime\\Results_Laila'
    xps = ['use_bottleneck', 'use_residual', 'nb_filters', 'depth', 'kernel_size', 'batch_size']

    mode = args.mode
    dataset_name = args.dataset
    do_train = args.train

    if mode == 'InceptionTime':
        classifier_name = 'inception'
        archive_name = ARCHIVE_NAMES[0]
        nb_iter_ = 5

        datasets_dict = read_all_datasets(root_dir, archive_name)

        for iter in range(nb_iter_):
            print('\t\titer', iter)

            trr = ''
            if iter != 0:
                trr = '_itr_' + str(iter)

            tmp_output_directory = os.path.join(root_dir, 'results', classifier_name, archive_name + trr)

            print(f'\t\t\tdataset_name: {dataset_name}')
            x_train, y_train, x_test, y_test, y_true, nb_classes, y_true_train, enc = prepare_data(datasets_dict, dataset_name)

            output_directory = tmp_output_directory + dataset_name + '\\'
            temp_output_directory = create_directory(output_directory)
            if temp_output_directory is None:
                print('Already_done', tmp_output_directory, dataset_name)
            else:
                input_shape = x_train.shape[1:]
                classifier = create_classifier(classifier_name, input_shape, nb_classes, output_directory)
                fit_classifier(classifier, x_train, y_train, x_test, y_test, y_true)
                print('\t\t\t\tDONE')
                create_directory(output_directory + '/DONE')
                classifier.model = load_model(output_directory + 'best_model.keras')
                extract_features(classifier, x_train, output_directory)

        # Ensembling after iterations
        classifier_name = 'nne'
        datasets_dict = read_all_datasets(root_dir, archive_name)
        tmp_output_directory = os.path.join(root_dir, 'results', classifier_name, archive_name + trr)

        print(f'\t\t\tdataset_name: {dataset_name}')
        x_train, y_train, x_test, y_test, y_true, nb_classes, y_true_train, enc = prepare_data(datasets_dict, dataset_name)

        output_directory = tmp_output_directory + dataset_name + '\\'

        input_shape = x_train.shape[1:]
        classifier = create_classifier(classifier_name, input_shape, nb_classes, output_directory, clf_name = 'inception')
        print("Training starting..")

        fit_classifier(classifier, x_train, y_train, x_test, y_test, y_true)
        print("Training completed")
        print('\t\t\t\tDONE')
    elif mode == 'InceptionTime_xp':
        archive_name = 'TSC'
        classifier_name = 'inception'
        max_iterations = 5

        datasets_dict = read_all_datasets(root_dir, archive_name)

        if do_train:
            for xp in xps:
                xp_arr = get_xp_val(xp)
                print('xp', xp)

                for xp_val in xp_arr:
                    print('\txp_val', xp_val)

                    kwargs = {xp: xp_val}

                    for iter in range(max_iterations):
                        trr = ''
                        if iter != 0:
                            trr = '_itr_' + str(iter)
                        print('\t\titer', iter)

                        output_directory = os.path.join(root_dir,'results',classifier_name,xp,str(xp_val),archive_name + trr,dataset_name)

                        print('\t\t\tdataset_name', dataset_name)
                        x_train, y_train, x_test, y_test, y_true, nb_classes, y_true_train, enc = prepare_data(datasets_dict, dataset_name)

                        temp_output_directory = create_directory(output_directory)
                        if temp_output_directory is None:
                            print('\t\t\t\tAlready_done')
                            continue

                        input_shape = x_train.shape[1:]

                        from classifiers import inception
                        classifier = inception.Classifier_INCEPTION(output_directory, input_shape, nb_classes,
                                                                    verbose=False, build=True, **kwargs)

                        classifier.fit(x_train, y_train, x_test, y_test, y_true)
                        create_directory(output_directory + '/DONE')

                        print('\t\t\t\tDONE')
        

        # Ensembling phase for xp experiments
        archive_name = ARCHIVE_NAMES[0]
        classifier_name = 'nne'

        datasets_dict = read_all_datasets(root_dir, archive_name)

        tmp_output_directory = os.path.join(root_dir, 'results', classifier_name, f'{archive_name}{trr}')

        for xp in xps:
            xp_arr = get_xp_val(xp)
            for xp_val in xp_arr:

                clf_name = 'inception/' + xp + '/' + str(xp_val)

                print(f'\t\t\tdataset_name: {dataset_name}')
                x_train, y_train, x_test, y_test, y_true, nb_classes, y_true_train, enc = prepare_data(datasets_dict, dataset_name)

                output_directory = os.path.join(tmp_output_directory, dataset_name)
                from classifiers import nne
                classifier = nne.Classifier_NNE(output_directory, x_train.shape[1:], nb_classes, clf_name=clf_name)

                classifier.fit(x_train, y_train, x_test, y_test, y_true)


    

    elif mode == 'run_length_xps':
        print("run_length_xps mode not implemented yet.")

    else:
        print(f"Unknown mode: {mode}")

if __name__ == "__main__":
    main()


