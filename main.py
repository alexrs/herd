from fastapi import FastAPI, Request, HTTPException, Depends
from pydantic import BaseModel
import asyncio
from typing import List, Dict
import uuid
import time
import datetime
import uvicorn
import argparse
import os
import json
import torch
from configparser import ConfigParser, ExtendedInterpolation
from herd.models import ModelValues, PathValues
from herd.router import Router
from herd.embeddings import Embeddings
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    StoppingCriteriaList,
    StoppingCriteria,
)
from sentence_transformers import SentenceTransformer
from peft import PeftModel
from functools import wraps
from loguru import logger
from contextlib import asynccontextmanager
from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env.

MODEL_LOCK = asyncio.Lock()

DEFAULT_STOPS = [
    "USER:",
    "ASSISTANT:",
    "### Instruction",
    "### Response",
    # These are often used as refusals, warnings, etc, but may also remove useful info.
    # "\nRemember,"
    # "\nPlease note,"
]

# TODO: What is this???
USER_STOP_TOKENS = [
    torch.tensor([3148, 1001, 29901], device="cuda"),
    torch.tensor([11889, 29901], device="cuda"),
]


class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = [stop for stop in stops + USER_STOP_TOKENS]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop) :])).item():
                return True
        return False


# TODO: Do not use global variables.

app_data = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model and adapters
    config = ConfigParser(interpolation=ExtendedInterpolation())
    config.read(app.args.config_file)

    model_values = ModelValues(**dict(config.items("Models")))
    path_values = PathValues(**dict(config.items("Paths")))

    # Create base_dir if it does not exists
    if not os.path.exists(path_values.base_dir):
        os.makedirs(path_values.base_dir)

    # Load tokenizer
    app_data["tokenizer"] = AutoTokenizer.from_pretrained(
        model_values.model, cache_dir=path_values.cache_dir
    )

    logger.debug(f"Loading model {model_values.model}")
    logger.debug(f"Tokenizer {app_data['tokenizer']}")

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    app_data["model"] = AutoModelForCausalLM.from_pretrained(
        model_values.model,
        load_in_4bit=True,
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
        device_map="auto",
        cache_dir=path_values.cache_dir,
    )

    if not app.args.only_base:
        embeddings_model = SentenceTransformer(
            model_values.embeddings_model, device="cuda"
        )
        embeddings_tokenizer = AutoTokenizer.from_pretrained(
            model_values.embeddings_model
        )
        embeddings = Embeddings(
            embeddings_model, embeddings_tokenizer, model_values.embeddings_max_length
        )

        # Read experts.json file
        with open(path_values.experts_file, "r") as json_file:
            experts = json.loads(json_file.read())
            # Create router
            app_data["router"] = Router(embeddings, experts)

            app_data["model"] = PeftModel.from_pretrained(
                app_data["model"],
                os.path.join(path_values.output_dir, "general"),
                adapter_name="general",
            )

            # Load adapters
            for expert_name in experts.keys():
                app_data["model"].load_adapter(
                    os.path.join(path_values.output_dir, expert_name), expert_name
                )
    yield
    app_data.clear()


app = FastAPI(lifespan=lifespan)


class ChatRequest(BaseModel):
    model: str
    experts: List[str] = None
    messages: List[Dict[str, str]]
    temperature: float = 0.5
    top_k: int = 50
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    stop: List[str] = DEFAULT_STOPS
    max_tokens: int = None
    top_experts: int = 1


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/v1/chat/completions")
async def chat_completions(raw_request: Request):
    """Simulate the OpenAI /v1/chat/completions endpoint.

    NOTE: Parameters supported in request include:
        - model: str. Ignored for now. Present for compatibility with OpenAI API.
        - messages: list[dict[str, str]]
        - temperature: float
        - repetition_penalty: float
        - top_p: float
        - top_k: int
        - stop: list[str]
        - max_tokens: int
        - top_experts: int. This parameter is not present in the OpenAI API.

    Example request:
    curl -s -XPOST http://127.0.0.1:8000/v1/chat/completions -H 'content-type: application/json' -d '{
      "model": "",
      "messages": [
        {
          "role": "system",
          "content": "A chat.",
        },
        {
          "role": "user",
          "content": "Lorem ipsum dolor sit amet"
        }
      ]
    }'
    """
    request = ChatRequest(**await raw_request.json())
    async with MODEL_LOCK:
        return complete_request(request)


def complete_request(request: ChatRequest):
    request_id = f"cmpl-{uuid.uuid4()}"

    stop_words_ids = [
        app_data["tokenizer"](stop_word, return_tensors="pt").input_ids.to("cuda")[0][
            1:
        ]
        for stop_word in request.stop
    ]
    stopping_criteria = StoppingCriteriaList(
        [StoppingCriteriaSub(stops=stop_words_ids)]
    )

    logger.debug(f"Request {request}")
    prompt = prompt_template(
        request.messages[0]["content"], request.messages[1]["content"]
    )
    logger.debug(f"Prompt {prompt}")
    logger.debug(f"Tokenizer {app_data['tokenizer']}")
    input_ids = app_data["tokenizer"](
        prompt, return_tensors="pt", truncation=True
    ).input_ids.cuda()
    # Route to expert
    if not app.args.only_base:
        expert, routing_duration = route_to_expert(request.messages[1]["content"], request.top_experts)
        logger.info(f"Routing to {expert} in {routing_duration} seconds")

    # Generate response
    response, duration = generate_response(
        input_ids, prompt, request, stopping_criteria
    )

    logger.debug(f"Response {response}")
    logger.debug(f"Duration {duration}")

    return {
        "id": request_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "duration": duration,
        "routing_duration": routing_duration,
        "model": request.model,
        "expert": expert,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response.strip(),
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": len(input_ids[0]),
            "completion_tokens": len(response[0]),
            "total_tokens": len(input_ids[0]) + len(response[0]),
        },
    }


def measure_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        started_at = datetime.datetime.utcnow()
        result = func(*args, **kwargs)
        duration = (datetime.datetime.utcnow() - started_at).total_seconds()
        return result, duration

    return wrapper


@measure_time
def route_to_expert(instruction: str, top: int = 1):
    # Experts is a list of tuples (expert_name, score).
    experts = app_data["router"].route(instruction, top)
    if top == 1:
        # If we only want the top expert, set it as an adapter
        app_data["model"].set_adapter(experts[0][0])

    else:
        # Otherwise, we compute a new adapter as a combination of the top experts.
        # We generate a unique name for the adapter because even if the same experts are used
        # the weights may be different.
        adapter_name = str(hash(datetime.datetime.now()))
        weights = [expert[1] for expert in experts]
        # Invert the weights to prioritize smaller weights
        inverted_weights = [1 / weight for weight in weights]

        # Calculate the new weights by normalizing the inverted weights
        w = [weight / sum(inverted_weights) for weight in inverted_weights]
        e = [expert[0] for expert in experts]
        logger.debug(f"Creating adapter for: {list(zip(e, w))}")

        app_data["model"].add_weighted_adapter(
            e,
            w,
            combination_type="linear",
            adapter_name=adapter_name,
        )
        app_data["model"].set_adapter(adapter_name)

    return experts


@measure_time
def generate_response(
    input_ids: torch.Tensor,
    prompt: str,
    request: ChatRequest,
    stopping_criteria: StoppingCriteriaList,
):
    max_tokens = (
        app_data["model"].config.max_position_embeddings - len(input_ids[0]) - 1
    )

    output = app_data["model"].generate(
        input_ids=input_ids,
        stopping_criteria=stopping_criteria,
        repetition_penalty=request.repetition_penalty,
        top_p=request.top_p,
        top_k=request.top_k,
        temperature=request.temperature,
        max_new_tokens=max_tokens,
        do_sample=True,
        # use_cache=False,
    )

    logger.debug(f"Output done")

    return app_data["tokenizer"].batch_decode(
        output.detach().cpu().numpy(), skip_special_tokens=True
    )[0][len(prompt) :]


def prompt_template(system: str, instruction: str):
    prompt = f"""{system}
### Input:
{instruction}

### Response:
"""
    return prompt


def main():
    parser = argparse.ArgumentParser(
        description="LMoE API server, somewhat similar to OpenAI API.",
    )
    parser.add_argument("-i", "--host", type=str, default="127.0.0.1", help="host name")
    parser.add_argument("-p", "--port", type=int, default=8000, help="port number")
    parser.add_argument("--config-file", default="config_experts.ini")
    parser.add_argument("--only-base", default=False, type=bool)

    args = parser.parse_args()
    app.args = args

    # Start the API server.
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
        timeout_keep_alive=5,
    )


if __name__ == "__main__":
    main()
