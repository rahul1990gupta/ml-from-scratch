import unittest 

from basic_search import (
    spell_correct,
    tfidf,
    build_index,
    pagerank
)

class TestBasicSearch(unittest.TestCase):
    def setUp(self):
        pass 

    def test_spell_check(self):
        data = [
            ("cot", "cat"),
            ("mankey" ,"monkey"),
            ("rhinocoros", "rhinoceros"),
            ("Alephant", "elephant"),
            ("jebra", "zebra")
        ]
        for ic, correct in data:
            self.assertEqual(spell_correct(ic), correct)

    def test_tfidf(self):
        documents = [
            "Elephants are one of the largest mammals on the planet, which live on the land.",
            "Elephants and Rhinoceros are most common largest land animal found in India.",
            "People have cats and monkeys as their pet all over the world. Cats are more common, but monkeys can be a great pet too.",
            "Zebras are wild and are mainly found in plains on Africa. Stripes on their body is their striking feature which can help in sppotting them from away.",
            "Rhinoceros weight about a ton and have skin that can even deflect bullets. However, they are very gentle creature and remain at peace if left alone."
        ]

        data = [
            ("elephant", [0, 1]),
            ("cat", [2]),
            ("zebra", [3]),
            ("monkey", [2]),
            ("rhinoceros", [1, 4])
        ]
        iix, tf = build_index(documents)

        for q, pages in data:
            self.assertListEqual(pages, tfidf(q, iix, tf, documents))

    def test_pagerank(self):
        documents = [
            ("elephants.md", "Elephants are one of the largest mammals on the planet, which live on the land."),
            ("mammals.md", "Elephants<elephants.md> and Rhinoceros<rhinoceros.md> are most common largest land animal found in India."),
            ("rhinoceros.md", "Rhinoceros weight about a ton and have skin that can even deflect bullets. However, they are very gentle creature just like elephants<elephants.md> and remain at peace if left alone."),
            ("migration.md", "Elephant<elephants.md> often migrate in herds from one corner of africa to another corner of the african continet in search of food and water.")
        ]
        
        edges = [
            (1, 0),
            (1, 2),
            (2, 0),
            (3, 0)
        ]
        n_nodes = 4
        scores = pagerank(n_nodes, edges)
        self.assertGreater(scores[0], max(scores[1:]))
        