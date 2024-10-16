import unittest 
from learning_to_rank import ndcg

class TestLearningToRank(unittest.TestCase):
    def setUp(self):
        pass 

    def test_ndcg(self):
        score_min = ndcg([0, 1, 2, 3, 3], 5)
        score_max = ndcg([3, 3, 2, 1, 0], 5)

        self.assertLess(score_min, score_max)


    def test_ranknet(self):
        pass 

    def test_lambdarank(self):
        pass 
