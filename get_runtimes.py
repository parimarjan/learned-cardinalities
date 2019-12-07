import pickle
import argparse
import glob
import pdb

def read_flags():
    parser = argparse.ArgumentParser()
    parser.add_argument("--join_dir", type=str, required=False,
            default="./join_results")
    return parser.parse_args()

def main():
    fns = glob.glob(args.join_dir + "/*.pkl")
    for fn in fns:
        with open(fn, "rb") as f:
            data = pickle.load(f)
        print(data.keys())
        pdb.set_trace()


args = read_flags()
main()
