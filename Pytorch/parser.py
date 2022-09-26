import argparse

parser = argparse.ArgumentParser("argument for training")
parser.add_argument("--r",default=2,type=int,help="33")
args=parser.parse_args()
print(args.r)
args.i=0
print(args.i)