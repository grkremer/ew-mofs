from Similarity import Similarity_of_Two_GOTerms, Similarity_of_Set_of_GOTerms
from goatools.obo_parser import GODag
from goatools.base import download_go_basic_obo
from goatools.obo_parser import GODag
import numpy as np

fin_dag = download_go_basic_obo("go-basic.obo")
go = GODag(fin_dag, optional_attrs={'relationship'}, load_obsolete=True)
def f(gen1,gen2):
    return Similarity_of_Two_GOTerms(gen1, gen2, go, 'GOGO')


