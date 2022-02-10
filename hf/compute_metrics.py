import argparse
from datasets import load_metric

def read(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines()]

def main():
    parser = argparse.ArgumentParser(description="Compute metrics (currently supports only BLEU)")
    parser.add_argument("--ref", required=True, help="Gold output file")
    parser.add_argument("--sys", required=True, help="Inferred output file")
    args = parser.parse_args()

    ref = read(args.ref)
    sys = read(args.sys)

    metric = load_metric("sacrebleu")
    eval_metric = metric.compute(predictions=sys, references=[[r] for r in ref], force=True)
    print("BLEU:", eval_metric["score"])


if __name__ == "__main__":
    main()
