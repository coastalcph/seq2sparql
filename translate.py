import json

from tqdm import tqdm
from google.cloud import translate_v2 as translate

translate_client = translate.Client()


def translate_with_qmark(q, target):
    q = q if q[-1] == "?" else q + "?"
    q_trans = translate_client.translate(q, target_language=target)["translatedText"]
    return q_trans[:-1] if q_trans[-1] == "?" else q_trans


def translate_file(target):
    with open(f"./cwq/dataset.json", "r") as f:
        data = json.load(f)
    for idx in tqdm(range(len(data))):
        data[idx][f"questionWithBrackets_{target}"] = translate_with_qmark(data[idx][f"questionWithBrackets"], target)
        data[idx][f"questionPatternModEntities_{target}"] = translate_with_qmark(data[idx][f"questionPatternModEntities"], target)
    with open(f"./cwq/dataset.json", "w") as f:
        json.dump(data, f)


def main():
    translate_file("kn")
    translate_file("he")
    translate_file("zh")


if __name__ == "__main__":
    main()
