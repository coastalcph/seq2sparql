import json
import re
from tqdm import tqdm
from google.cloud import translate_v2 as translate

translate_client = translate.Client()

def replacer(s, newstring, index, nofail=False):
    # raise an error if index is outside of the string
    if not nofail and index not in range(len(s)):
        raise ValueError("index outside given string")

    # if not erroring, but the index is still not in the correct range..
    if index < 0:  # add it to the beginning
        return newstring + s
    if index > len(s):  # add it to the end
        return s + newstring

    # insert the new string between "slices" of the original
    return s[:index] + newstring + s[index + 1:]

def translate_with_qmark(q, qMod, target):
    front_bracket = ["[", "(", "{", "«", "「", "『", "【", "'"]
    back_bracket = ["]", ")", "}", "»", "」", "』", " 】", "'"]
    q = q if q[-1] == "?" else q + "?"
    qBrackets = q
    num_f = 0
    num_b = 0
    #replace parenthesis
    for position, char in enumerate(q):
        if char=="[":
            qBrackets = replacer(qBrackets, front_bracket[num_f], position)
            num_f += 1
        if char=="]":
            qBrackets = replacer(qBrackets, back_bracket[num_b], position)
            num_b += 1
    assert num_f == num_b
    
    #get the ModEntities in order
    #modEntity_list = []
    #for position, char in enumerate(qMod):
    #    if char=="M":
    #        if qMod[position+1].isnumeric():
    #            if position+2 != len(qMod):
    #                if qMod[position+2] == " ":
    #                    modEntity_list.append(qMod[position:position+2])
    #            else:
    #                modEntity_list.append(qMod[position:position+2])
    try:
        #assert len(modEntity_list) == num_f
        #translate
        q_trans = translate_client.translate(qBrackets, target_language=target)["translatedText"]
        q_trans_me = translate_client.translate(qMod, target_language=target)["translatedText"]
        #for i in range(0, len(modEntity_list)):
        #    front = front_bracket[i]
        #    back = back_bracket[i]
        #    mod = modEntity_list[i] 
        #    start = q_trans_me.find(front)
        #    end = q_trans_me.find(back)
        #    q_trans_me = f"{q_trans_me[:start]}{mod}{q_trans_me[end+1:]}"      
        #print(f"qBrackets: {qBrackets}")
        #print(f"q_trans_me: {q_trans_me}")

        q_trans_bracket = q_trans[:-1] if q_trans[-1] == "?" else q_trans
        q_trans_me = q_trans_me[:-1] if q_trans_me[-1] == "?" else q_trans_me
        return q_trans_bracket, q_trans_me
    except Exception as e:
        print("* An Entity was unbracketed. Skipped example.")
        return "NOPE", "NOPE"

def translate_file(target):
    with open(f"./cwq/dataset.json", "r") as f:
        data = json.load(f)
    nope = 0
    for idx in tqdm(range(len(data))):
        qBrackets, qModEntities =  translate_with_qmark(data[idx][f"questionWithBrackets"],data[idx][f"questionPatternModEntities"], target)
        if qBrackets == "NOPE":
            nope+=1
        data[idx][f"questionWithBrackets_{target}"] = qBrackets
        data[idx][f"questionPatternModEntities_{target}"] = qModEntities 
    print(f"Num fucky: {nope}")
    with open(f"./cwq/dataset_paranthesis.json", "w") as f:
        json.dump(data, f)


def main():
    translate_file("kn")
    translate_file("he")
    translate_file("zh")


if __name__ == "__main__":
    main()
