import pandas as pd
import openai
from tqdm import tqdm

# Configure local Ollama client with OpenAI-compatible API endpoints
openai.api_key = 'ollama'
openai.api_base = "http://localhost:11434/v1"

data = pd.read_csv('../dataset/reveal/reveal_slice_train.csv')
data_1 = data['text'].values.tolist()
data_2 = data['label'].values.tolist()
data2 = pd.read_csv('../dataset/reveal/interp_train.csv')
data_3 = data2['text'].values.tolist()
data_4 = data2['label'].values.tolist()
k = len(data_3)

q = (
    "Task: Provide a high-density, technical summary of the C/C++ function for security embedding.\n"
    "Constraints: Strictly under 250 words. Use professional auditing language.\n"
    "Use plain text only, avoid markdown formatting and special symbols like * or #.\n"
    "Structure:\n"
    "1. [Security Analysis]: Identify vulnerabilities. Pay extra attention to resource management (memory leaks, unclosed handles), "
    "boundary conditions, and type safety. Even if the logic seems standard, evaluate it against strict CWE standards.\n"
    "2. [Purpose]: Brief overview of the function's goal.\n"
    "3. [Logic]: Precise step-by-step execution flow.\n\n"
    "IMPORTANT: Keep it concise to avoid truncation. Focus on data flow and memory safety.\n\n"
    "Code:\n"
)
dict = {}

LOCAL_MODEL = "qwen3-coder:30b"

# Connection test for local LLM deployment
try:
    test_response = openai.ChatCompletion.create(
        model=LOCAL_MODEL,
        messages=[{"role": "user", "content": "Test message"}],
        timeout=30
    )
    print(f"Connection test to local Ollama ({LOCAL_MODEL}) succeeded.")
except Exception as e:
    print(f"Ollama connection error: {e}")
    print("Please check if the Ollama service is running and the model name is correct.")
    exit()

print(f"Initial k: {k}, Data length: {len(data_1)}")
if k >= len(data_1):
    print("Checkpoint k has reached or exceeded data length. Execution terminated.")
    exit()

pbar = tqdm(range(k, len(data_1)), desc="Generating Explanations", unit="it")

for i in pbar:
    # Generate technical code summaries via local LLM
    rsp = openai.ChatCompletion.create(
        model=LOCAL_MODEL,
        messages=[
            {"role": "system",
             "content": "You are a security expert who provides high-density, concise code summaries for downstream ML models."},
            {"role": "user", "content": q + data_1[i]}
        ],
        max_tokens=512,
        temperature=0.7
    )
    msg = rsp.get("choices")[0]["message"]["content"]

    # Post-processing: remove formatting tokens and clean whitespaces
    p = msg.replace('\n', ' ')
    while '  ' in p:
        p = p.replace('  ', ' ')
    p = p.strip()
    p = p.replace('*', ' ')
    p = p.replace('#', ' ')

    data_3.append(p)
    data_4.append(data_2[i])

    # Intermittent checkpoint saving
    if (i + 1) % 10 == 0 or (i + 1) == len(data_1):
        dict['text'] = data_3
        dict['label'] = data_4
        df = pd.DataFrame(dict)
        df.to_csv('../dataset/reveal/interp_train.csv', index=None)

    pbar.set_postfix({"current_k": i + 1, "checkpoint_saved": (i + 1) % 10 == 0})

print("Data processing completed and safely stored.")