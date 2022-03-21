import os
import datetime
import logging

import pandas as pd
import numpy as np

from ..utils import get_args_parser, cur_time

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class DataLoader:
    def __init__(self, path, seed=100):
        """
        :param path: str
        """
        self.num_features = None
        self.feature_names = None
        self.num_pairs = None
        self.num_sessions = None
        self.df = None
        self.path = path
        self.pickle_path = path[:-3] + 'pkl'
        self.seed = seed

    def _load_mslr(self):
        logger.info(cur_time() + " load file from {}".format(self.path))
        df = pd.read_csv(self.path, sep=" ", header=None)
        df.drop(columns=df.columns[-1], inplace=True)
        self.num_features = len(df.columns) - 2
        self.num_paris = None
        logger.info(cur_time() + " finish loading from {}".format(self.path))
        logger.info("dataframe shape: {}, features: {}".format(df.shape, self.num_features))
        return df

    def _parse_data(self, df):
        """
        :param df: pandas.DataFrame
        :return: pandas.DataFrame
        """
        logger.info(cur_time() + " parse dataframe ... {}".format(df.shape))
        for col in range(1, len(df.columns)):
            if ':' in str(df.iloc[:, col][0]):
                df.iloc[:, col] = df.iloc[:, col].apply(lambda x: x.split(":")[1])
        df.columns = ['rel', 'qid'] + [str(f) for f in range(1, len(df.columns) - 1)]

        for col in [str(f) for f in range(1, len(df.columns) - 1)]:
            df[col] = df[col].astype(np.float32)

        logger.info(cur_time() + " finish parsing dataframe")
        self.df = df
        self.num_sessions = len(df.qid.unique())
        return df

    def get_num_pairs(self):
        if self.num_pairs is not None:
            return self.num_pairs
        self.num_pairs = 0
        for _, Y in self.process_query_batch(self.df):
            Y = Y.reshape(-1, 1)
            pairs = Y - Y.T
            pos_pairs = np.sum(pairs > 0, (0, 1))
            neg_pairs = np.sum(pairs < 0, (0, 1))
            assert pos_pairs == neg_pairs
            self.num_pairs += pos_pairs + neg_pairs
        return self.num_pairs

    def process_raw_data(self):
        """
        :return: pd.DataFrame
        """
        if os.path.isfile(self.pickle_path):
            logger.info(cur_time() + " load from pickle file {}".format(self.pickle_path))
            self.df = pd.read_pickle(self.pickle_path)
            self.num_features = len(self.df.columns) - 2
            self.num_sessions = len(self.df.qid.unique())
        else:
            self.df = self._parse_data(self._load_mslr())
            self.df.to_pickle(self.pickle_path)
        self.feature_names = ['{}'.format(i) for i in range(1, self.num_features + 1)]
        return self.df

    def generate_batch(self, df, batchsize=10000):
        """
        :param df: pandas.DataFrame, contains column qid
        :param batchsize: size of a batch
        :returns: numpy.ndarray qid, rel, x_i
        """
        idx = 0
        while idx * batchsize < df.shape[0]:
            r = df.iloc[idx * batchsize: (idx + 1) * batchsize, :]
            yield r.qid.values, r.rel.values, r[self.feature_names].values
            idx += 1

    def process_query_batch(self, df=None):
        """
        :param df:
        :return: X features, y rel
        :rtype: numpy.ndarray, numpy.ndarray
        """
        if df is None:
            df = self.df
        qids = df.qid.unique()
        np.random.seed(self.seed)
        np.random.shuffle(qids)
        for qid in qids:
            df_qid = df[df.qid == qid]
            yield df_qid[self.feature_names].values, df_qid.rel.values


if __name__ == "__main__":
    parser = get_args_parser()
    parser.add_argument("--data_path", type=str, default="../data/mslr-web10k/")
    args = parser.parse_args()
    dataloader = DataLoader(args.data_path)
    data = dataloader.process_raw_data()
    print(data.head())
    print(data.shape)



