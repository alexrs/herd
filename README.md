# Herd
A group of Llamas.

Leverage Mixture of Expert (MoE) models for large language models (LLMs) and enhance their performance with advanced PEFT methods.

## Overview
This project is part of an MSc thesis focused on exploring MoE models for LLMs. By leveraging PEFT methods like LoRA and QLoRA, it seeks to offer an effective solution to use a base model in combination with an extensive set of adapters and a method to identify the appropriate adapter.

The basic idea behind this is to come up with a solution that allows for using a base model + a large set of adapters + a method to find the right adapter.

## Key Features
- **Prompt-Expert Mapping**: Determines the right expert based on the input prompt's distance from expert centroids.
- **Combination of Adapters**: Allows merging multiple adapters based on input prompt proximity to each expert's centroid.


## Inspiration & Credits
The first iteration of this project is heavily based on [airoboros/lmoe](https://github.com/jondurbin/airoboros/tree/main/airoboros/lmoe). Additionally,
I have also included the option to combine multiple adapters according to the distance between the input prompt and the centroid for each expert (and it seems like [@aicrum had a similar idea](https://twitter.com/aicrumb/status/1681846805959528448)).

## Experts & Segmentation
The experts are fine-tuned using QLoRA on the [jondurbin/airoboros-2.1 dataset](https://huggingface.co/datasets/jondurbin/airoboros-2.1/viewer/default/train) using the same segmentation as in the original project.

 Expert Name | Categories |
-------------|------------|
 qa          | quiz, multiple_choice, contextual, counterfactual_contextual |
 creative    | card, writing, experience, song, roleplay, gtkm, rp, detailed_writing, joke |
 code        | coding |
 reasoning   | cot, theory_of_mind, riddle, orca |
 function    | agent, plan |
 general     | wordgame, trivia, general |


### Fine-tuning Experts
To fine-tune:

```sh
python scripts/main.py finetune
```

### Computing Expert-Prompt Distance
To compute the distance between the input prompt and each expert we:

1. Sample a few instructions from each expert and compute the average embedding for each expert.
2. Save the average embedding for each expert as a numpy array in `experts/`

When a new prompt is received, we:
1. Load the average embedding for each expert.
2. Use [faiss](https://github.com/facebookresearch/faiss) to compute the distance between the input prompt and each expert.

## API Documentation
Herd provides a simple REST API. It is based on OpenAI's API.


```py
python scripts/app.py
```

The following options can be used:
- `--port (-p)`: The port to run the server on. Default: `8000`
- `--host (-i)`: The host to run the server on. Default: `127.0.0.1`
- `--config-file`: The config file to use. Default: `config.ini`
- `--only-base`: Only use the base model. Default: `False`


### Querying the Model

To query the model we can run:
```sh
curl -s -XPOST http://127.0.0.1:8000/v1/chat/completions -H 'content-type: application/json' -d '{
    "model": "herd",
    "messages": [
        {
          "role": "system",
          "content": "A chat."
        },
        {
          "role": "user",
          "content": "Lorem ipsum dolor sit amet"
        }
      ]
    }'
```

The following options can be passed:
- `model` (str): The name of the model.
- `messages` (List[Dict[str, str]]): The list of messages in the chat.
- `temperature` (float, optional): The temperature for generating responses. Defaults to 0.5.
- `top_k` (int, optional): The number of top-k tokens to consider. Defaults to 50.
- `top_p` (float, optional): The cumulative probability for generating responses. Defaults to 1.0.
- `repetition_penalty` (float, optional): The repetition penalty for generating responses. Defaults to 1.0.
- `stop` (List[str], optional): The list of stop words. Defaults to DEFAULT_STOPS.
- `max_tokens` (int, optional): The maximum number of tokens in the response. Defaults to None.
- `top_experts` (int, optional): The number of top experts to consider. Defaults to 1.


### Segment Alpaca
https://colab.research.google.com/drive/1nPxDLt0ExCLLx3j-VovRPDJR14ET17no?usp=sharing
