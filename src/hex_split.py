"""
@author: Pierre Obermaier
"""
import argparse
import sys
from functools import reduce

import numpy as np

import block_reader

SIZE_MB = 1000000


class HexSplitter:

    def __init__(self, inp, out, analysis, max_size):
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

        self.analysis_only = analysis
        self.header_injected = False
        self.header = self.extract_header()
        analysed = False
        for self.features in block_reader.blocks(self.input, int(max_size) * SIZE_MB):
            self.features = np.array(list(map(lambda x: x.split('\t'), self.features)), dtype=object)
            if not analysed:
                self.hex_features = self.analyse(self.features[0])
            self.extract_features(self.hex_features)
            self.output.write('\t'.join(self.header))
            for row in self.features:
                self.output.write('\t'.join(map(lambda x: str(x), row)))

    def extract_header(self):
        return np.array(self.input.readline().split('\t'))

    def analyse(self, features):
        def analyse_feature(feature_name, max_value):
            bits = (len(max_value) - 2) * 4
            return feature_name, np.where(self.header == feature_name)[0][0], bits

        hex_features = list(map(lambda x: analyse_feature(self.header[x[0]], x[1]),
                                filter(lambda x: self.is_hex(x[1]), enumerate(features))))
        if self.analysis_only:
            self.output.write('Potential candidates for hex transformation:\nFeature\tPosition\tBit Size\n')
            self.output.write(reduce(lambda x, y: x + '{}\t{}\t{}\n'.format(y[0], y[1], y[2]), hex_features, ''))
            sys.exit(0)
        return hex_features

    def extract_features(self, features):
        shift = 0
        for feature in features:
            feature_name, pos, bit_size = feature
            pos += shift
            if not self.header_injected:
                self.inject_headers(pos + 1, bit_size, feature_name)
            self.inject_features(pos + 1, bit_size)
            self.populate_features(pos, bit_size)
            shift += bit_size
        self.header_injected = True

    def inject_headers(self, pos, column_count, base_name):
        self.header = np.insert(self.header, pos, list(map(lambda x: base_name + '_' + str(x), range(column_count))))

    def inject_features(self, pos, column_count):
        self.features = np.insert(self.features, [pos],
                                  np.zeros(shape=(len(self.features), column_count), dtype=np.int), axis=1)

    def populate_features(self, hex_value_pos, bits):
        for i in range(len(self.features)):
            hex_value = int(self.features[i][hex_value_pos], 16)
            bit = 1
            for j in range(bits):
                if hex_value & bit > 0:
                    self.features[i][hex_value_pos + j + 1] = 1
                bit = bit << 1

    @staticmethod
    def is_hex(value):
        if '0x' in value:
            try:
                int(value, 16)
                return True
            except ValueError:
                pass
        return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Processes input splitting hex represented features into multiple '
                                                 'numerical features and writing them into the output.\n\nTo be able to'
                                                 ' extract the features and set new ones the the input MUST contain a '
                                                 'header with the feature names in its first row.\nGenerated features '
                                                 'will be named {original_feature_name}_{pos} where pos is the position'
                                                 ' of the corresponding bit in the hexadecimal representation.',
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-a', '--analysis', action='store_true', default=False,
                        help='prints a list of features represented by hex values\nand the amount of bits (features) '
                             'needed to represent them')
    parser.add_argument('-m', '--max-size', default=512,
                        help='the maximum amount of data (in MB) to be read from the input at once\n default is 500MB')
    parser.add_argument('-i', '--input', dest='inp', default=sys.stdin,
                        help='path to the input to be read from\nThe default is standard input')
    parser.add_argument('-o', '--output', dest='out', default=sys.stdout,
                        help='path to the output file to be written to\nThe default is standard output')
    arg_dict = vars(parser.parse_args())
    HexSplitter(**arg_dict)
