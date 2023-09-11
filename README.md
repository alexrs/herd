# Herd
A group of Llamas.

## Experts


## Fine-Tuning models

## LoRA Adapters

## API

```sh
curl -s -XPOST http://127.0.0.1:8000/v1/chat/completions -H 'content-type: application/json' -d '{
    "model": "??",
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

