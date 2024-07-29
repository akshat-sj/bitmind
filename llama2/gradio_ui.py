import gradio as gr
import torch
from transformers import LlamaTokenizer
from modeling_llama_amd import LlamaForCausalLM
import os
import gc
import time
import qlinear
from utils import Utils
import io
import logging
from typing import List, Dict

log_capture_string = io.StringIO()
ch = logging.StreamHandler(log_capture_string)
ch.setLevel(logging.CRITICAL)
logging.getLogger().addHandler(ch)

from rank_bm25 import BM25Okapi
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
nltk.download('punkt')

def load_model():
    tokenizer = LlamaTokenizer.from_pretrained("./llama-2-wts-hf/7B_chat")
    ckpt = "pytorch_llama27b_w_bit_4_awq_amd.pt"
    model = torch.load(ckpt)
    Utils.print_model_size(model)
    _ = gc.collect()
    model.eval()
    model = model.to(torch.bfloat16)
    print(model)
    return model, tokenizer


model, tokenizer = load_model()
dev = os.getenv("DEVICE")
torch.set_num_threads(8)

for n, m in model.named_modules():
    if isinstance(m, qlinear.QLinearPerGrp):
        print(f"Preparing weights of layer : {n}")
        m.device = "aie"
        m.quantize_weights()

print(model)
Utils.print_model_size(model)

def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    return [word.lower() for word in word_tokenize(text) if word.isalnum() and word.lower() not in stop_words]

def load_document(file_name):
    with open(file_name, 'r', encoding='utf-8') as file:
        return file.read()

def retrieve_relevant_passages_bm25(query: str, document: str, top_k: int = 3, max_chars: int = 1000) -> str:
    # Split the document into paragraphs
    paragraphs = document.split('\n\n')
    
    # Preprocess paragraphs
    tokenized_corpus = [preprocess_text(para) for para in paragraphs]
    
    # Create BM25 model
    bm25 = BM25Okapi(tokenized_corpus)
    
    # Tokenize query
    tokenized_query = preprocess_text(query)
    
    # Get scores
    scores = bm25.get_scores(tokenized_query)
    
    # Get top k paragraph indices
    top_indices = scores.argsort()[-top_k:][::-1]
    
    relevant_text = ""
    for i in top_indices:
        if len(relevant_text) + len(paragraphs[i]) <= max_chars:
            relevant_text += paragraphs[i] + " "
        else:
            remaining_chars = max_chars - len(relevant_text)
            relevant_text += paragraphs[i][:remaining_chars] + "..."
            break
    
    return relevant_text.strip()

def generate_text(prompt, max_new_tokens, option):
    global model, tokenizer
   
    # Clear previous logs
    log_capture_string.truncate(0)
    log_capture_string.seek(0)
    selected_file = ""
    # Load the selected document
    if option == "DOOM":
        selected_file = "docs/doom.txt"
    if option == "Pokemon Red":
        selected_file = "docs/red.txt"
    document = load_document(selected_file)
   
    # Retrieve relevant passages with BM25
    relevant_passages = retrieve_relevant_passages_bm25(prompt, document, top_k=3, max_chars=1000)
   
    # Construct augmented prompt
    augmented_prompt = f"Context: {relevant_passages}\n\nQuestion: {prompt}\nAnswer:"
   
    # Tokenize input
    inputs = tokenizer(augmented_prompt, return_tensors="pt", truncation=True, max_length=512)
    input_ids = inputs.input_ids.to(model.device)
    attention_mask = inputs.attention_mask.to(model.device)
   
    start_time = time.time()
    generate_ids = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=max_new_tokens)
    generate_time = time.time() - start_time
    prompt_tokens = input_ids.shape[1]
    total_tokens = generate_ids.shape[1]
    new_tokens = total_tokens - prompt_tokens
   
    # Decode the entire output
    full_response = tokenizer.decode(generate_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
    
    # Extract only the answer part
    answer_start = full_response.find("Answer:") + len("Answer:")
    response = full_response[answer_start:].strip()
    
    tokens_per_second = new_tokens / generate_time if generate_time > 0 else 0
    performance_info = f"Total tokens: {total_tokens}, New tokens: {new_tokens}, Tokens/sec: {tokens_per_second:.2f}"
   
    # Capture logs
    log_contents = log_capture_string.getvalue()
   
    return response, performance_info, log_contents

# Gradio interface
iface = gr.Interface(
    fn=generate_text,
    inputs=[
        gr.Textbox(lines=2, label="Prompt"),
        gr.Slider(minimum=1, maximum=100, value=30, step=1, label="Max New Tokens"),
        gr.Dropdown(choices=["DOOM", "Pokemon Red"], label="Select Text File")
    ],
    outputs=[
        gr.Textbox(label="Generated Text"),
        gr.Textbox(label="Performance Metrics"),
        gr.Textbox(label="Model Logs", lines=10)
    ],
    title="LLaMA 2 Text Generation with BM25 Retrieval",
    description="Enter a prompt to generate text using the quantized LLaMA 2 model with BM25 retrieval from selected text file."
)

# Launch the interface
if __name__ == "__main__":
    iface.launch()