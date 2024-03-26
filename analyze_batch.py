# -*- coding: utf-8 -*-
# @Time    : 19/03/24 14:30 
# @Author  : Burooj Ghani
# @Affiliation  : Naturalis Biodiversity Center
# @Email   : burooj.ghani at naturalis.nl
# @File    : analyze_batch.py


from models import *
from config import *


# I/O args
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--slist', type=str, default='inputs/list_sp_ml.csv', help='Path to the species list.')
parser.add_argument('--flist', type=str, default='inputs/species_list_nl.csv', help='Path to the filter list of species.')
parser.add_argument('--i', type=str, default='audio', help='Input audio sample.')
parser.add_argument('--o', type=str, default='tmp/avesecho', help='Output directory for temporary audio chunks.')
parser.add_argument('--mconf', type=float, default=None, help='Minimum confidence threshold for predictions.')
parser.add_argument('--lat', type=float, default=None, help='Latitude for geographic filtering.')
parser.add_argument('--lon', type=float, default=None, help='Longitude for geographic filtering.')
parser.add_argument('--add_filtering', action='store_true', help='Enable geographic filtering.')
parser.add_argument('--add_csv', action='store_true', help='Save predictions to a CSV file.')
parser.add_argument('--maxpool', action='store_true', help='Use model for generating temporally-summarised output.')
parser.add_argument('--model_name', type=str, default='fc', help='Name of the model to use.')


if __name__ == "__main__":

    args = parser.parse_args()
   
    start_time = time.time()

    classifier = AvesEcho(model_name=args.model_name, slist=args.slist, flist=args.flist,
                    add_filtering=args.add_filtering, mconf=args.mconf,
                    outputd = args.o, avesecho_mapping='inputs/list_AvesEcho.csv',
                    maxpool=args.maxpool, add_csv=args.add_csv, args=args)
    
    classifier.analyze_batch(args.i, lat=args.lat, lon=args.lon)

    # Compute the elapsed time in seconds
    elapsed_time = time.time() - start_time
            
    # Print the time it took
    print(f"It took {elapsed_time:.2f}s to analyze the batch.")
    
    