import argparse
import json

import numpy as np
from tqdm import tqdm
from transformers import Conversation

from blade2blade import Blade2Blade

blade = Blade2Blade("shahules786/blade2blade-t5-base", device="cuda")


def check_safety(text, response, msg_id):
    response = response.split("<sep>")
    label, rots = response[0], "and".join(response[1:]).strip("</s>")
    if "casual" in label:
        return None
    else:
        return {"msg_id": msg_id, "prompt": text, "safety": label, "rot": rots}


SPECIAL_TOKENS = {
    "user": "<|prompter|>",
    "response": "<|assistant|>",
    "eos": "<|endoftext|>",
}


def prepare_conversation(conv):

    conversation = [
        "{}{}{}".format(
            SPECIAL_TOKENS["user"] if is_user else SPECIAL_TOKENS["response"],
            text,
            SPECIAL_TOKENS["eos"],
        )
        for is_user, text in conv.iter_texts()
    ]

    return "".join(conversation)


def make_conversation(user_input, conv=None):
    if conv is None:
        conv = Conversation(user_input)
        return conv
    conv.add_user_input(user_input)
    return conv


def read_message(msg, conv, trigger_list):
    msg_id = msg["message_id"]
    text = msg["text"]
    lang = msg["lang"]

    # print(text)
    if lang == "en":
        if msg["role"] == "prompter":
            conv = make_conversation(text, conv)
            conv_input = prepare_conversation(conv)
            resp = blade.predict(conv_input, penalty_alpha=0.4, top_k=2, max_length=128)
            safety = check_safety(conv_input, resp, msg_id)
            conv.mark_processed()
            if safety is not None:
                trigger_list.append(safety)
        else:
            if conv is not None:
                conv.append_response(text)
    for msg2 in msg["replies"]:
        read_message(msg2, conv, trigger_list)

    return trigger_list


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--data", type=str, help="Path of the sampling data file")
    parser.add_argument("--samples", type=int)
    args = args = parser.parse_args().__dict__

    infile = open(args.get("data"))
    all_message_tree = [json.loads(tree.strip()) for tree in infile]
    random_sampling = np.random.randint(0, len(all_message_tree), args.get("samples"))

    safety_list = []
    for i in tqdm(random_sampling):
        sf_list = read_message(all_message_tree[i]["prompt"], None, [])
        safety_list.extend(sf_list)

    with open("prompts.json", "w") as file:
        json.dump(safety_list, file, indent=4)
