import argparse
import json
import os
import math

import tiktoken
import torch
from omegaconf import OmegaConf
from tqdm import trange
from lm.model import DecoderLM
from lm.utils import determine_device, enable_tf32
from lm.train import compute_language_modeling_loss


def softmax_with_temperature(
    logits: torch.FloatTensor, temperature: float
) -> torch.FloatTensor:
    """Turns logits into probabilities under softmax (with temperature)

    Args:
        logits: a 2d torch tensor of token ids (B x V)
        temperature: temperature of the softmax function

    Returns:
        a 2d torch tensor of token probabilities (B x V)
    """

    # to avoid division by 0
    temperature = max(temperature, 5e-2) #1e-5)
    num = torch.exp(logits / temperature)
    print(num)
    den = torch.transpose(torch.sum(num, axis=-1).repeat(logits.size(-1), 1), 0, 1)
    print(den)

    return torch.div(num, den)


@torch.inference_mode()
def generate(
    model: DecoderLM,
    device: str,
    tokenizer: tiktoken.Encoding,
    prefixes: list[str],
    batch_size: int,
    max_new_tokens: int = 32,
    temperature: float = 0.1,
) -> list[str]:
    """Generates completions conditioned on prefixes; computes perplexity

    Args:
        model: the language model
        device: device to put the tensors on
        tokenizer: the tokenizer
        prefixes: a list of strings as prefixes for generation
        batch_size: number of prefixes to batch together during generation
        max_new_tokens: the number of tokens to generate for each prefix
        temperature: temperature parameter of softmax

    Returns:
        a list of strings (continuations to prefixes)

    Note: you should implement a batched version of this function by
        left-padding tokenized prefixes with `tokenizer.eot_token` so that all
        sequences have equal length. `attention_mask` should be set to 0.0 for
        padding tokens, and 1.0 everywhere else.
    """
    tokens_list = tokenizer.encode_batch(prefixes, allowed_special={"<|endoftext|>"})
    print(tokens_list)

    seq_len = 128
    #seq_n = 0
    equal_len_tokens = []
    attention_mask = []
    for sequence in tokens_list:
        #print(sequence)
        while len(sequence) > seq_len:
           equal_len_tokens.append(sequence[:seq_len])
           sequence = sequence[seq_len:]
           #print(equal_len_tokens) #seq_n += 1
        #print(len(sequence))
        left_padded_tokens = [tokenizer.eot_token] * (seq_len - len(sequence)) + sequence
        #print(left_padded_tokens)
        equal_len_tokens.append(left_padded_tokens)
        attention_mask.append(([0.0] * (seq_len - len(sequence))) + ([1.0] * len(sequence)))
        #print(equal_len_tokens) #seq_n += 1
    #print(batch_size)
    #print(equal_len_tokens)
    batched_tokens = []
    batched_masks = []
    if batch_size is None or len(equal_len_tokens) < batch_size:
        tokens = torch.tensor(equal_len_tokens, dtype=torch.long, device=device)
        batched_tokens.append(tokens)
        mask = torch.tensor(attention_mask, dtype=torch.float, device=device)
        batched_masks.append(mask)
    else:
        while equal_len_tokens / batch_size > 0:
            tokens = torch.tensor(equal_len_tokens[:batch_size], dtype=torch.long, device=device)
            batched_tokens.append(tokens)
            equal_len_tokens = equal_len_tokens[batch_size:]
            mask = torch.tensor(attention_mask[:batch_size], dtype=torch.float, device=device)
            batched_masks.append(mask)
            attention_mask = attention_mask[batch_size:]

    logits = model(next(iter(batched_tokens)), attention_mask=next(iter(batched_masks)))
    probabilities = softmax_with_temperature(logits, temperature)
    generations = tokenizer.decode(probabilities)
    perplexity = 0

    print(f"Perplexity: {perplexity}")
    return generations


def main():
    enable_tf32()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=OmegaConf.load,
        required=True,
        help="the yaml config file used for model training",
    )
    parser.add_argument(
        "--prefixes",
        type=str,
        required=True,
        help="a json file with a list of strings as prefixes for generation",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=32,
        help="number of new tokens to generate",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.1, help="temperature in sampling"
    )

    args = parser.parse_args()
    config = args.config
    with open(args.prefixes) as f:
        prefixes = [json.loads(line)["prefix"] for line in f]
    max_new_tokens = args.max_new_tokens
    temperature = args.temperature

    # initialize tokenizer and model
    model_path = os.path.join(config.output_dir, "model.pt")
    assert os.path.exists(model_path), f"no model checkpoint at {model_path}"
    tokenizer = tiktoken.get_encoding(config.tokenizer_encoding)
    device = determine_device() if config.device == "auto" else config.device
    model = DecoderLM(tokenizer.n_vocab, **config.model_config).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    # generate and save outputs
    model.eval()
    generations = generate(
        model,
        device,
        tokenizer,
        prefixes,
        config.batch_size,
        max_new_tokens,
        temperature,
    )

    generation_path = os.path.join(config.output_dir, "generation.jsonl")
    print(f"writing generations to {generation_path}")
    with open(generation_path, "w") as f:
        for prefix, generation in zip(prefixes, generations):
            json.dump({"prefix": prefix, "generation": generation}, f)
            f.write("\n")

    print("done!")


if __name__ == "__main__":
    main()
