# pyright: basic
import argparse

from satire.dataset.utils import dump_sentences

parser = argparse.ArgumentParser("CLI tools for dataset")
subparsers = parser.add_subparsers(dest="command", required=True)

generate_parser = subparsers.add_parser(
    "dump",
    help="Dump the sentences of a csv dataset",
)
generate_parser.add_argument(
    "dataset_path",
    type=str,
    help="Path to the input csv dataset",
)
generate_parser.add_argument(
    "output_path",
    type=str,
    help="Path to the output text file",
)

args = parser.parse_args()

if args.command == "dump":
    dump_sentences(args.dataset_path, args.output_path)
    print(f"Dumped {args.dataset_path} to {args.output_path}")
