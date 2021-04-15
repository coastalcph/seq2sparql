import json
from google.cloud import translate_v2 as translate

translate_client = translate.Client()


def translate_file(target, split):
    with open(f"./multicfq/multicfq_{split}.json", "r") as f:
        data = json.load(f)
    for idx in range(len(data)):
        data[idx][f"questionWithBrackets_{target}"] = translate_client.translate(
            data[idx][f"questionWithBrackets"], target_language=target
        )["translatedText"]
    with open(f"./multicfq/multicfq_{split}.json", "w") as f:
        json.dump(data, f)


def main():
    # translate_file("kn", "test")
    # translate_file("he", "test")
    translate_file("zh", "test")


if __name__ == "__main__":
    main()
