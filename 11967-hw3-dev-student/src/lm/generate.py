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
    print(logits.shape)
    temperature = max(temperature, 5e-2) #1e-5)
    num = torch.exp(logits / temperature)
    #print(num)
    den = torch.transpose(torch.sum(num, axis=-1).repeat(logits.size(-1), 1), 0, 1)
    #print(den)

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
    #print(tokens_list)
    #print(max(tokens_list))
    seq_len = 128
    #seq_n = 0
    equal_len_tokens = []
    attention_mask = []
    for sequence in tokens_list:
        #print(sequence)
        while len(sequence) >= seq_len:
           equal_len_tokens.append(sequence[:seq_len])
           sequence = sequence[seq_len:]
           #print(equal_len_tokens) #seq_n += 1
        #print(len(sequence))
        # Make [tokenizer.eot_token]
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
        tokens = torch.tensor(equal_len_tokens, dtype=torch.long)
        batched_tokens.append(tokens)
        mask = torch.tensor(attention_mask, dtype=torch.float)
        batched_masks.append(mask)
    else:
        while len(equal_len_tokens) / batch_size > 0:
            tokens = torch.tensor(equal_len_tokens[:batch_size], dtype=torch.long)
            batched_tokens.append(tokens)
            equal_len_tokens = equal_len_tokens[batch_size:]
            mask = torch.tensor(attention_mask[:batch_size], dtype=torch.float)
            batched_masks.append(mask)
            attention_mask = attention_mask[batch_size:]
    #print(len(batched_tokens))
    #print(len(batched_tokens[0]))
    #if len(batched_tokens[-1]) < batch_size:
        # Make 0 tokenizer.eot_token
    #    batched_tokens[-1] = torch.cat((batched_tokens[-1], torch.full((batch_size - len(batched_tokens[-1]), seq_len), tokenizer.eot_token)))
    #    batched_masks[-1] = torch.cat((batched_masks[-1], torch.full((batch_size - len(batched_masks[-1]), seq_len), 0.0)))
    batch_of_tokens = next(iter(batched_tokens)).to(device)
    #print(batch_of_tokens.shape)
    batch_of_masks = next(iter(batched_masks)).to(device)
    #print(batch_of_masks.shape)
    new_logits = model(batch_of_tokens) #, attention_mask=batch_of_masks)
    generations = []
    for sequence in new_logits:
        print("Sequence shape in generate")
        print(sequence.shape)
        probabilities = softmax_with_temperature(sequence[-1*max_new_tokens - 1: -1, :], temperature)
        #print(probabilities.shape)
        new_token_encodings = torch.multinomial(probabilities, num_samples=1)
        #print(new_tokens.shape)
        generated_tokens = []
        generation = ""
        for new_token_encoding in new_token_encodings:
            new_token = tokenizer.decode(new_token_encoding.tolist())
            generated_tokens.append(new_token)
            generation += new_token
        print(generation)
        generations.append(generation)
        #loss_per_sequence = compute_language_modeling_loss(,) #torch.nn.functional.cross_entropy(new_token_encodings, sequence[-1*max_new_tokens:-1, :], reduction="none")
        #print("Loss for " + generation + ": " + str(loss_per_sequence))

    loss = compute_language_modeling_loss(batch_of_tokens, new_logits)

    perplexity = math.exp(loss)
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
