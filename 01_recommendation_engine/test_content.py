import unittest
from content import run


class TestContent(unittest.TestCase):
    def setUp(self):
        pass
    
    @unittest.skip("To Be Run")
    def test_top_5_genres(self):
        top5 = run("genre")
        
        self.assertListEqual(
            top5, 
            ['Action', 'Comedy', ' Sci-Fi', ' Shounen', ' Fantasy']                        
        )

    @unittest.skip("To Be Run")
    def test_rank_by_member_rating(self):
        anime_ids = [137, 127, 296, 82, 1723]
        
        ranked_ids = run("rank", anime_ids)
        expected_ranked_ids = [1723, 82, 296, 127, 137]
        self.assertListEqual(expected_ranked_ids, ranked_ids)

    @unittest.skip("To Be Run")
    def test_candidate_list_in_same_genre(self):
        anime_ids = [1723]

        recs = run("candidate_same_genre", anime_ids)
        self.assertEqual(len(recs), 3204)

    @unittest.skip("To Be Run")   
    def test_candidate_list_with_matching_names_kw(self):
        anime_ids = [1723]

        recs = run("candidate_matching_names", anime_ids)
        self.assertEqual(len(recs), 363)
    
    @unittest.skip("To Be Run")   
    def test_candidate_list_with_knn(self):
        anime_ids = [1723]

        recs = run("candidate_knn", anime_ids)
        self.assertEqual(len(recs), 23)
    
