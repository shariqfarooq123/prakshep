from extractor import Extractor
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("filename",help="filename of the image containing the card")
parser.add_argument("-a","--aadhaar",help="card is aadhaar (default is pan)",action="store_true")
parser.add_argument("-v","--verbose",help="enable logging",action="store_true")
args = parser.parse_args()
if args.aadhaar:
    card_type='a'
else:
    card_type='p'
verbose=0
if args.verbose:
    print "verbose enabled"
    verbose=1
ext = Extractor(card_type,verbose=verbose)
data = ext.extract(args.filename)
print data

