import keras
import numpy as np
from utils.utils import calculate_metrics
from utils.utils import create_directory
from utils.utils import check_if_file_exits
import gc
from utils.constants import UNIVARIATE_ARCHIVE_NAMES  as ARCHIVE_NAMES
import time
import os


class Classifier_NNE:
    def create_classifier(self, model_name, input_shape, nb_classes, output_directory, verbose=False,
                          build=True):
        if self.check_if_match('inception.*', model_name):
            from classifiers import inception
            return inception.Classifier_INCEPTION(output_directory, input_shape, nb_classes, verbose,
                                                  build=build)
        print(f"No classifier matched for model_name: {model_name}")
        return None

    def check_if_match(self, rex, name2):
        import re
        pattern = re.compile(rex)
        return pattern.match(name2)

    def __init__(self, output_directory, input_shape, nb_classes, verbose=False, nb_iterations=5,
                 clf_name='inception', archive_name = None):
        self.classifiers = ['inception']
        out_add = ''

        for cc in self.classifiers:
            out_add = out_add + cc + '-'
        self.archive_name = archive_name if archive_name else ARCHIVE_NAMES[0]            #previously: self.archive_name = ARCHIVE_NAMES[0]
        self.iterations_to_take = [i for i in range(nb_iterations)]
        for cc in self.iterations_to_take:
            out_add = out_add + str(cc) + '-'
        self.output_directory = output_directory.replace('nne', os.path.join('nne', out_add))#previously: self.output_directory = output_directory.replace('nne','nne' + '\\' + out_add)
        print('self.output_directory:', self.output_directory)
        for cc in self.classifiers:
            if cc is None:
                cc = ''
            out_add = out_add + cc + ''
        create_directory(self.output_directory)
        self.dataset_name = os.path.basename(os.path.dirname(output_directory)) #self.dataset_name = output_directory.split('/')[-2]
        self.verbose = verbose
        self.models_dir = output_directory.replace('nne', 'classifier')

    def fit(self, x_train, y_train, x_test, y_test, y_true):
        # no training since models are pre-trained
        start_time = time.time()

        y_pred = np.zeros(shape=y_test.shape)

        ll = 0

        # loop through all classifiers
        for model_name in self.classifiers:
            print(f"Using model_name: {model_name}")
            print("output_directory:", self.output_directory)
            print("self.models_dir:", self.models_dir)
            print("model_name:", model_name)
            print("self.archive_name:", self.archive_name)
            # loop through different initialization of classifiers
            for itr in self.iterations_to_take:
                if itr == 0:
                    itr_str = ''
                else:
                    if 'itr_' in self.archive_name:
                        itr_str = ''
                    else:
                        itr_str = '_itr_' + str(itr)

                curr_archive_name = self.archive_name + itr_str
                print("curr_archive_name:", curr_archive_name)

                curr_dir = self.models_dir.replace('classifier', model_name)

                model = self.create_classifier(model_name, None, None,
                                               curr_dir, build=False)

                predictions_file_name = curr_dir + 'y_pred.npy'
                # check if predictions already made
                if check_if_file_exits(predictions_file_name):
                    # then load only the predictions from the file
                    curr_y_pred = np.load(predictions_file_name)
                else:
                    # then compute the predictions
                    curr_y_pred = model.predict(x_test, y_true, x_train, y_train, y_test,
                                                return_df_metrics=False)
                    keras.backend.clear_session()

                    np.save(predictions_file_name, curr_y_pred)

                y_pred = y_pred + curr_y_pred

                ll += 1

        #average predictions
        y_pred = y_pred / ll

        # save predictions
        np.save(os.path.join(self.output_directory, 'y_pred.npy'), y_pred)

        #convert the predicted from binary to integer
        y_pred = np.argmax(y_pred, axis=1)

        duration = time.time() - start_time
        df_metrics = calculate_metrics(y_true, y_pred, duration)
        df_metrics.to_csv(self.output_directory + 'df_metrics.csv', index=False)
        gc.collect()
