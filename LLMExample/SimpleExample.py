from ctransformers import AutoModelForCausalLM

llm = AutoModelForCausalLM.from_pretrained(
    "zoltanctoth/orca_mini_3B-GGUF", model_file="orca-mini-3b.q4_0.gguf"
)

# Very specific prompt
prompt = "What is the name of the capital city of India? She asked. Please only respond with the city name and then stop talking. IA answers:"

# Text is printed suddenly 
# print(prompt + llm(prompt))

# Text is printed in parts like in ChatGPT, and without the prompt
for word in llm(prompt, stream=True):
    print(word, end="", flush=True)
print()