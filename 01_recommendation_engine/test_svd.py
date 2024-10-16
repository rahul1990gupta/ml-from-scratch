import pandas as pd 
import unittest 
from mysvd import get_algo

class TestSVD(unittest.TestCase):
    def setUp(self):
        pass 

    @unittest.skip("Later")
    def test_all_10(self):
        
        data = {
            "userid": [i//3 + 1 for i in range(9)],
            "itemid": [ i%3 + 1 for i in range(9)],
            "ratings": [10 for _ in range(9)]
        }
        df = pd.DataFrame.from_dict(data)
        algo = get_algo(df)

        self.assertEqual(algo.trainset.global_mean, 10)
        
        # self.assertListEqual(algo.bi, [10, 10, 10])
        # self.assertListEqual(algo.bu, [10, 10, 10])

        self.assertAlmostEqual(algo.bi.mean(), 0.0, 1)
        self.assertAlmostEqual(algo.bu.mean(), 0.0, 1)

    @unittest.skip("Later")
    def test_one_low_rating(self):
        
        data = [
            (1, 1, 1),
            (1, 2, 10),
            (1, 3, 10),
            (2, 1, 1),
            (2, 2, 10),
            (2, 3, 10),
            (3, 1, 1),
            (3, 2, 10),
            (3, 3, 10),
        ]

        df = pd.DataFrame.from_records(data, columns=["userid", "itemid", "rating"])
        
        algo = get_algo(df)

        self.assertEqual(algo.trainset.global_mean, 7)
        
        self.assertLess(algo.bi[0], -1)
        self.assertGreater(algo.bi[1], 0.5)
        self.assertGreater(algo.bi[2], 0.5)
        
        self.assertAlmostEqual(algo.bu.mean(), 0.00, 1)


        # preds = []
        # for i in range(1, 4):
        #     for j in range(1, 4):
        #         pred = (i, j, algo.predict(i,j).est)
        #         preds.append(pred)
        #         print(pred)

    @unittest.skip("Later")
    def test_another(self):
        data = [
            (1, 1, 2),
            (1, 2, 3),
            (1, 3, 4),
            (2, 1, 3),
            (2, 2, 4),
            (2, 3, 5),
            (3, 1, 4),
            (3, 2, 5),
            (3, 3, 6),
        ]

        df = pd.DataFrame.from_records(data, columns=["userid", "itemid", "rating"])
        
        
        algo = get_algo(df)

        self.assertEqual(algo.trainset.global_mean, 4)
        
 
        self.assertLess(algo.bi[0], 0)

        self.assertGreater(algo.bi[2], 0)
        
        for i in range(3):
            self.assertAlmostEqual(algo.bi[i], algo.bu[i], 1)
    