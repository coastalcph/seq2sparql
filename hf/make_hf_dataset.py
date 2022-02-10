import os
import sys
import json

def read(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines()]

def write(output, lang, split):
    opath = os.path.join(all_output_path, "all_langs", mcd_type) if not lang else output_path.format(lang)
    os.makedirs(opath, exist_ok=True)
    opath = os.path.join(opath, f"{split}.json")
    with open(opath, "w", encoding="utf-8") as f:
        for line in output:
            f.write(json.dumps(line) + "\n")

def main():
    for split in ["train", "dev", "test"]:
        all_output = []
        for lang in ["en"]: # "zh", "he", "kn"
            src = read(encode_path.format(lang, split, split))
            tgt = read(decode_path.format(lang, split, split))
            assert len(src) == len(tgt), "Length mismatch."
            
            output = []
            for s, t in zip(src, tgt):
                output.append({"translation": {"src": s, "tgt": t}})
            write(output, lang, split)
            all_output.extend(output)
        write(all_output, None, split)


if __name__ == "__main__":
    mcd_type = sys.argv[1]
    encode_path = "../t2t_data/cwq_{}/" + mcd_type + "/lstm_seq2seq_attention/{}/{}_encode.txt"
    decode_path = "../t2t_data/cwq_{}/" + mcd_type + "/lstm_seq2seq_attention/{}/{}_decode.txt"
    output_path = "../data/hf_data/cwq_{}/" + mcd_type
    all_output_path = "../data/hf_data"
    main()