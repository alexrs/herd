from peft import get_peft_model, AutoPeftModelForCausalLM
from peft.utils import load_peft_weights
import datasets
from herd.finetune_utils import prepare_tokenizer
import os
import torch
import torch.nn.functional as F
from herd.prompter import Prompter


def prepare_model(pretrained_model_name_or_path, cache_dir):
    model = AutoPeftModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path,
        load_in_4bit=True,
        torch_dtype=torch.bfloat16,
        quantization_config=None,
        device_map="auto",
        cache_dir=cache_dir,
        is_trainable=False,
    )
    return model


activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = F.softmax(output, dim=-1)
    return hook


def check_experts_loaded(model, experts, experts_dir):
    expert_weights = []
    print(experts.keys())
    for expert in experts.keys():
        expert_weights.append(load_peft_weights(os.path.join(experts_dir, expert)))

    for i,e in enumerate(expert_weights):
        for name, param in e.items():
            if "lora_A" in name or "lora_B" in name:
                assert torch.allclose(param, model.state_dict()[name + ".default"][i:i+1])


def check_activations(model, tokenizer, dataset, experts, experts_dir):
    # we check the activations of the last router layer

    for name, module in model.named_modules():
        if "lora_router" in name:
            module.register_forward_hook(get_activation(name))
    # model.model.model.layers[-1].mlp.down_proj.lora_router.default.register_forward_hook(get_activation('lora_router'))

    # get a random sample from the dataset
    for i in range(0, len(experts)):
        samples = dataset.shuffle().filter(lambda row: row['cluster'] == i).select(range(20))

        # get the input ids
        prompter = Prompter()

        sum_per_sample = []
        activations = []
        for sample in samples:
            prompt = prompter.generate_prompt(sample, use_output=False)
            input_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).input_ids.cuda()
            outputs = model.generate(
                input_ids=input_ids,
                max_new_tokens=500,
                do_sample=True,
                top_p=0.9,
                temperature=0.9,
            )
            # print(
            #     f"Response: \n{tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(prompt):]}"
            # )

            activations.append(activation)

            # Assuming all tensors in the dictionary have the same shape
            tensor_shape = next(iter(activation.values())).shape
            total_sum = torch.zeros(tensor_shape, device='cuda:0')

            for _, tensor in activation.items():
                total_sum += tensor

            # print("Element-wise Total Sum:", total_sum)
            sum_per_sample.append(total_sum)


        # calculate total sum across all samples
        tensor_shape = next(iter(sum_per_sample)).shape
        total_sum = torch.zeros(tensor_shape, device='cuda:0')
        for sum in sum_per_sample:
            total_sum += sum

        # print(activations)

        print(f"Total Sum for i {i}: {total_sum}")


def test_molora(model_values, path_values, config, experts) -> None:
    experts_path = os.path.join(path_values.output_dir, "molora")
    model_path = os.path.join(experts_path, "all")
    model = prepare_model(model_path, path_values.cache_dir)
    # print(model)
    # model.load_experts(output_dir, 'default', list(experts.keys()), True)

    dataset = datasets.load_dataset(model_values.dataset, split="train")
    tokenizer = prepare_tokenizer(model_values.model, path_values.cache_dir)

    # check_experts_loaded(model, experts, experts_path)
    check_activations(model, tokenizer, dataset, experts, experts_path)

    # model.fc0.conv2.register_forward_hook(get_activation('fc0.conv2'))
    # model.fc1.conv2.register_forward_hook(get_activation('fc1.conv2'))

    # output = model(x)
    # print(activation['fc0.conv2'])
    # print(activation['fc0.conv1'])
