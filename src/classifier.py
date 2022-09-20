"""
@author: Pierre Obermaier
"""
import argparse
import pickle
import sys

import numpy as np

import block_reader
import ml_data

SIZE_MB = 1000000


class Classifier:

    def __init__(self, model, inp, out, header, single, batch):
        if isinstance(inp, str):
            raise NotImplementedError('Processing from a file is currently not supported. Use stdin to pass the input '
                                      'instead.')
        else:
            self.input = inp
        if isinstance(out, str):
            raise NotImplementedError('Writing to a file is currently not supported. It is written to stdout from where'
                                      'you may redirect it to a file.')
        else:
            self.output = out
        self.print_header = header
        self.model = self.load_model(model)
        self.feature_indices = self.pre_process()
        if batch is not None:
            for data in block_reader.blocks(self.input, int(batch) * SIZE_MB):
                self.process_all(data)
        elif single:
            self.process()
        else:
            self.process_all(self.input.readlines())

    def load_model(self, model_path):
        with open(model_path, 'rb') as fp:
            model = pickle.load(fp)
        return model

    def pre_process(self):
        header = self.input.readline()
        if self.print_header:
            self.output.write(header)

        feature_indices = self.get_features_indices(header)
        return feature_indices

    def process(self):
        for line in self.input:
            Y = self.model.predict([self.extract_features(line, self.feature_indices)])
            if Y == [1]:
                self.output.write(line)

    def process_all(self, data):
        lines = np.array(data)
        Y = self.model.predict(list(map(lambda x: self.extract_features(x, self.feature_indices), lines)))
        indices = filter(lambda x: x > -1, map(lambda x: x[0] if x[1] else -1, enumerate(Y)))
        for i in indices:
            self.output.write(lines[i])

    def extract_features(self, line, feature_indices):
        feature_array = line.split('\t')
        return np.array(list(map(lambda x: feature_array[x], feature_indices)))

    def get_features_indices(self, header):
        all_features = header.split('\t')
        features = ml_data.all_lite_features
        categorical_features = ml_data.tranalyzer_categorical_features
        non_categorical_features = [item for item in features if item not in categorical_features]
        # for the final feature set this list might be extracted and hardcoded, if it can be guaranteed that a feature
        # will always be at the same index
        return list(map(lambda x: all_features.index(x), non_categorical_features[:15]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process input using the given model to classify the data. Predicted '
                                                 'positives will be written to output.\n\nTo be able to extract the '
                                                 'features correctly the input MUST contain a header with the feature '
                                                 'names in its first row.\nBy default the data is processed in multi '
                                                 'mode, meaning everything from the input is read into memory and '
                                                 'processed at once.\nThis makes the whole processing a lot faster than'
                                                 ' single mode, but has potentially unlimited memory consumption.',
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('model', help='the model used for prediction')
    parser.add_argument('-s', '--single', default=False, action='store_true',
                        help='single mode: unlike multi mode only a single line of the input is processed at once which'
                             ' has the benefit of pretty much constant memory consumption, but the drawback of much '
                             'worse performance')
    parser.add_argument('-b', '--batch', default=None,
                        help='batch mode: like multi mode but only given amount (in MB) of data of the input is read at'
                             ' once to ensure that large files can be processed sufficiently fast even if the do not '
                             'fit into the memory at once')
    parser.add_argument('-H', '--header', default=False, action='store_true',
                        help='print the header present in the input to output')
    parser.add_argument('-i', '--input', dest='inp', default=sys.stdin,
                        help='path to the input to be read from\nThe default is standard input')
    parser.add_argument('-o', '--output', dest='out', default=sys.stdout,
                        help='path to the output file to be written to\nThe default is standard output')
    args_dict = vars(parser.parse_args())
    Classifier(**args_dict)
