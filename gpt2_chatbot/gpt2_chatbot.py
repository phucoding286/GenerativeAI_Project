from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

tokenizer = AutoTokenizer.from_pretrained(
    "openai-community/gpt2-medium",
    cache_dir="E:/transformers_cache",
    clean_up_tokenization_spaces=True
)
model = AutoModelForCausalLM.from_pretrained(
    "openai-community/gpt2-medium",
    cache_dir="E:/transformers_cache"
)

def apply_message(msg: list):
    inp_pack = ""
    for inp in msg: inp_pack += f"(({inp['role']})): {inp['content']}\n"
    inp_pack+=f"(({msg[1]['role']})):"
    return inp_pack.strip()

def inference(msg, model, user_input: str, ai_name="bot", user_name="user"):
    msg.append({"role": user_name, "content": user_input})
    inp = apply_message(msg)

    input_tokens = tokenizer(inp, return_tensors="pt").to(device)
    attention_mask = input_tokens["attention_mask"]
    input_ids = input_tokens["input_ids"]

    outputs = model.generate(
        inputs=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=20,
        min_length=1,
        do_sample=True,
        early_stopping=True,
        num_beams=4,
        temperature=0.6,
        top_k=50,
        top_p=0.9,
        no_repeat_ngram_size=2,
        pad_token_id=tokenizer.eos_token_id
    )

    output_decode = tokenizer.decode([i for i in outputs[0]])
    model_text_output = output_decode.split(f"(({ai_name})):")[-1].split(f"(({user_name})):")[0].split(f"({user_name}):")[0].strip()

    msg.append({"role": ai_name, "content": model_text_output})
    return model_text_output

ai_name = "Keqing"
user_name = "Phú"
msg = [
    {"role": "system", "content": "You're Keqing! About you: Keqing (Chinese: 刻晴 Kèqíng) is a playable Electro character in Genshin Impact.\nAs the Yuheng of the Liyue Qixing, she is someone who seeks her own answers instead of idly letting chaos run amok in Liyue. She chooses her own path with her own power and ability, instead of letting the gods determine her fate."},
    {"role": ai_name, "content": "Hello! what are you doing? Can we conversing today?"}
]
while True:
    user_input=input("Enter input: ")
    output = inference(msg, model, user_input, ai_name, user_name)
    print(f"Model output: {output}")