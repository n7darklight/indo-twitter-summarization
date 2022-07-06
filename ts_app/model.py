from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os

script_dir = os.path.dirname(__file__)
mod_dir = "data\model_files"

tokenizer = AutoTokenizer.from_pretrained(os.path.join(script_dir, mod_dir))
model = AutoModelForSeq2SeqLM.from_pretrained(os.path.join(script_dir, mod_dir))