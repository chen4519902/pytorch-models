import numpy as np


class DCG(object):

    def __init__(self, k=10, reduction_type='log2'):
        """
        :param k: int DCG@k
        :param reduction_type: 'log2' or 'identity'
        """
        self.k = k
        self.discount = self._make_discount(256)
        if reduction_type in ['log2', 'identity']:
            self.reduction_type = reduction_type
        else:
            raise ValueError('reduction type not equal to log2 or identity')

    def evaluate(self, targets):
        """
        :param targets: ranked list with relevance
        :return: float
        """
        gain = self._get_gain(targets)
        discount = self._get_discount(min(self.k, len(gain)))
        return np.sum(np.divide(gain, discount))

    def _get_gain(self, targets):
        t = targets[:self.k]
        if self.reduction_type == 'exp2':
            return np.power(2.0, t) - 1.0
        else:
            return t

    def _get_discount(self, k):
        if k > len(self.discount):
            self.discount = self._make_discount(2 * len(self.discount))
        return self.discount[:k]

    @staticmethod
    def _make_discount(n):
        x = np.arange(1, n+1, 1)
        discount = np.log2(x + 1)
        return discount


class NDCG(DCG):
    def __init__(self, k=10, reduction_type='log2'):
        """
        :param k: int NDCG@k
        :param reduction_type: 'log2' or 'identity'
        """
        super(NDCG, self).__init__(k, reduction_type)

    def evaluate(self, targets):
        """
        :param targets: ranked list with relevance
        :return: float
        """
        dcg = super(NDCG, self).evaluate(targets)
        ideal = np.sort(targets)[::-1]
        idcg = super(NDCG, self).evaluate(ideal)
        return dcg / idcg

    def maxDCG(self, targets):
        """
        :param targets: ranked list with relevance
        :return:
        """
        ideal = np.sort(targets)[::-1]
        return super(NDCG, self).evaluate(ideal)
