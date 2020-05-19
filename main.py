import math

from gensim import corpora
from gensim import models, similarities
from data_preprocessing import Tokenizer
import numpy as np
import matplotlib.pyplot as plt
import random
import logging
logger = logging.getLogger('my_module_name')
logger.propagate = False
from lsa_models_pack import Corpus, TFIDFModel, LSIModel



if __name__ == "__main__":

    print("###########################################\n"
          "LATENT SEMANTIC ANALYSIS MODEL WELCOMES YOU\n"
          "###########################################")

    restart = True
    documents = []

    documents_query = [
        "Laptop Apple Macbook Pro 13 Retina 256GB (MV962UA/A) Space Gray",
        "Laptop Apple MacBook Pro 16 1TB 2019 (MVVM2UA/A) Silver",
        "Laptop Apple MacBook Pro 13 Retina 128Gb (MUHQ2UA/A) Silver",
        "Laptop Apple MacBook Pro 16 256GB 2019 (MVVM2UA/A) Silver",
        "New Apple MacBook Pro (16-Inch, 16GB RAM, 512GB Storage) - Space Gray",
        "New Apple MacBook Pro (13-inch, 8GB RAM, 256GB SSD Storage, Magic Keyboard) - Space Gray",
        "New Apple MacBook Pro (16-Inch, 16GB RAM, 512GB Storage) - Space Gray",
        "Apple MacBook Pro 13 Retina Space Gray with Touch Bar (MUHP2) 2019",
        "Apple MacBook Pro 16 Retina Space Gray with Touch Bar (MVVJ2) 2019",
        "Apple MacBook Pro 16 Retina Silver with Touch Bar (MVVM2) 2019",
        "13-inch MacBook Pro Touch Bar 1.4GHz quad-core 8th-gen Intel i5 128GB",
        "Apple MacBook Pro 13 Retina Silver with Touch Bar (MUHQ2) 2019",
        "Apple MacBook Pro 16 Retina Space Gray with Touch Bar Custom (Z0Y0001H4) 2019",
        "Apple MacBook Pro w/ Touch Bar 13.3 - Space Grey (Intel Core i5 1.4GHz/128GB SSD/8GB RAM) ",
        "Apple MacBook Pro w/ Touch Bar 13.3 - Space Grey (Intel Core i5 1.4GHz/256GB SSD/8GB RAM) ",
        "Apple MacBook Pro (2020) w/ Touch Bar 13.3 - Space Grey (Intel i5 1.4GHz / 256GB SSD / 8GB RAM) "
    ]

    documents_little = [
        "Laptop Apple Macbook Pro 13 Retina 256GB (MV962UA/A) Space Gray",
        "Laptop Apple MacBook Pro 16 1TB 2019 (MVVM2UA/A) Silver",
        "Laptop Apple MacBook Pro 13 Retina 128Gb (MUHQ2UA/A) Silver",
        "Laptop Apple MacBook Pro 16 256GB 2019 (MVVM2UA/A) Silver"
    ]

    document_laptops = [
        "Dell G5 15 5590 (G5590FI716S2H1D206L-9BK) Deep Space Black",
        "Laptop Apple Macbook Pro 13 Retina 256GB (MV962UA/A) Space Gray",
        "Dell Inspiron G5 15 5590 (5590G5i716S2H1R26-WBK) Black",
        "Laptop Apple MacBook Pro 16 1TB 2019 (MVVM2UA/A) Silver",
        "Dell Inspiron G7 17 7790 (G7790FI916S5D2080W-9Gr) Gray",
        "Laptop Apple MacBook Pro 13 Retina 128Gb (MUHQ2UA/A) Silver",
        "Dell XPS 13 7390 (X3716S4NIW-64S) Platinum Silver",
        "Laptop Apple MacBook Pro 16 256GB 2019 (MVVM2UA/A) Silver",
        "Dell Inspiron G3 17 3779 (G37581NDL-60B) Black",
        "Dell Vostro 15 5590 (N5104VN5590EMEA01_2005_UBU_Rail-08) Urban Gray",
        "Dell Inspiron 15 3593 (I3558S2NIL-75B) Black"
    ]

    document_fairytales = [
        "What the frog had said came true, and the queen had a little girl who was so pretty that the king could not contain himself for joy, and ordered a great feast.",
        "He invited not only his kindred, friends and acquaintances, but also the wise women, in order that they might be kind and well-disposed towards the child. ",
        "There were thirteen of them in his kingdom, but, as he had only twelve golden plates for them to eat out of, one of them had to be left at home.",
        "The feast was held with all manner of splendor and when it came to an end the wise women bestowed their magic gifts upon the baby - one gave virtue, another beauty, a third riches, and so on with everything in the world that one can wish for.",
        "The first little pig built his house out of straw because it was the easiest thing to do. ",
        "The second little pig built his house out of sticks. ",
        "The third little pig built his house out of bricks."
    ]

    document_fairytales_2 = [
        "The first little pig built his house out of straw because it was the easiest thing to do. ",
        "The second little pig built his house out of sticks. ",
        "The third little pig built his house out of bricks."
    ]

    # ########################################################

    while restart:
        flag = input("\nType: \n 'C' - for computing classification\n"
                     " 'Q' - for computing query-similarity\n > ")

        if flag.lower() == 'c':
            flag_c = input("\n Great! Which topic of classification you"
                           " want to compute (type number):\n"
                           " 1 - Laptops\n"
                           " 2 - Fairy-tails\n > ")
            if int(flag_c) == 1:
                documents = document_laptops
            elif int(flag_c) == 2:
                documents = document_fairytales
        elif flag.lower() == 'q':
            documents = document_query
	else:
	    restart = False

        tokenizer = Tokenizer(documents)
        texts = tokenizer.tokenize()

        initial_corpus = Corpus(texts)
        initial_corpus.create_corpus()

        tfidf_model = TFIDFModel(initial_corpus)
        tfidf_model.apply_tfidf()
        tfidf_model.tfidf_corpus()

        lsi_model = LSIModel(tfidf_model, documents)

        if flag.lower() == 'c':
            lsi_model.get_double_classification()

        elif flag.lower() == 'q':
            lsi_model.apply_lsi(num_of_topics=2)
            # Laptop Apple MacBook Pro 16 256GB 2019 (MVVM2UA/A) Silver
            lsi_model.similar_queries(top=5)

        flag_maybe_end = input("\n Type: \n 'Y' - to continue playing with a model\n 'N' - to finish the program\n > ")
        if flag_maybe_end.lower() == 'n':
            restart = False

    print("\n\n Bye bye!")
