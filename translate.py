import json
from google.cloud import translate_v2 as translate

translate_client = translate.Client()


def translate_file(target):
    with open(f"./cwq/dataset.json", "r") as f:
        data = json.load(f)
    for idx in range(len(data)):
        data[idx][f"questionWithBrackets_{target}"] = translate_client.translate(
            data[idx][f"questionWithBrackets"], target_language=target
        )["translatedText"]
        data[idx][f"questionPatternModEntities_{target}"] = translate_client.translate(
            data[idx][f"questionPatternModEntities"], target_language=target
        )["translatedText"]
    with open(f"./cwq/dataset.json", "w") as f:
        json.dump(data, f)


def main():
    translate_file("kn")
    translate_file("he")
    translate_file("zh")


if __name__ == "__main__":
    main()
