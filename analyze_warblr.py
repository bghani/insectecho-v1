# -*- coding: utf-8 -*-
# @Time    : 19/03/24 14:30 
# @Author  : Burooj Ghani
# @Affiliation  : Naturalis Biodiversity Center
# @Email   : burooj.ghani at naturalis.nl
# @File    : analyze_warblr.py


from models import *
from config import *

       
if __name__ == "__main__":

    classifier = AvesEcho(model_name='fc', slist='inputs/list_sp_ml.csv', flist='species_list.csv',
                    add_filtering=True, mconf=0.1,
                    outputd = 'tmp/avesecho', avesecho_mapping='inputs/list_AvesEcho.csv',
                    maxpool=False, add_csv=False, args=None)
    
    species, probabilities = classifier.analyze_warblr_audio('audio/malta_test.wav', lat=None, lon=None)

    print(f'Predictions for {species} are: {probabilities}')


