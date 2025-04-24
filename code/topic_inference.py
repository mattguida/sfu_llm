from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import numpy as np 
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams
from pydantic import BaseModel, conlist, Field

from collections import defaultdict
from typing import List, Dict, Optional
import jsonlines as jl
import json
import os
import sys 
import torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prompts.prompts import PROMPT_TOPIC
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

torch.cuda.empty_cache()

class Output(BaseModel):
    article_id: Optional[str] = None
    comment_id: Optional[str] = None
    topic: str

class TopicClassifier:
    def __init__(self, model_name: str, prompt: str, article_output_file: str, comment_output_file: str, cache_dir: str = None, dtype: str = "bfloat16"):
        self.model_name = model_name
        self.prompt = prompt
        self.article_output_file = article_output_file
        self.comment_output_file = comment_output_file
        self.cache_dir = cache_dir
        self.dtype = dtype
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=self.cache_dir)
        self.llm = LLM(model=self.model_name, dtype=self.dtype, trust_remote_code=True, quantization="bitsandbytes", load_format="bitsandbytes")
        
    def load_data(self, file_paths: List[str], data_type: str):
        data = []
        for path in file_paths:
            df = pd.read_csv(path)
            if data_type == 'article':
                article_df = df[['article_id', 'title', 'article_text']].drop_duplicates(subset=['article_id'])
                data.extend(article_df.to_dict(orient="records"))
            elif data_type == 'comment':
                comment_df = df[['comment_id', 'comment_text', 'article_id']]
                data.extend(comment_df.to_dict(orient="records"))
        return data
        
    def classify(self, file_paths: List[str], output_schema: BaseModel, data_type: str):
        all_data = self.load_data(file_paths, data_type) 
        json_schema = output_schema.model_json_schema()
        guided_decoding_params = GuidedDecodingParams(json=json_schema)
        sampling_params = SamplingParams(guided_decoding=guided_decoding_params, max_tokens=4000)

        output_file = self.article_output_file if data_type == 'article' else self.comment_output_file
        
        with open(output_file, "w") as f:  
            for item in all_data:
                if data_type == 'article':
                    content = f"Title: {item['title']}\n\nContent: {item['article_text']}"
                    unique_id = str(item['article_id'])
                elif data_type == 'comment':
                    content = f"Comment: {item['comment_text']}"
                    unique_id = str(item['comment_id'])

                prompt = self.prompt.replace("{unique_id}", str(unique_id)).replace("{content_type}", data_type)
                messages = [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": content},
                ]
                text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

                outputs = self.llm.generate([text], sampling_params)
                print(f"Processing ID: {unique_id} of type: {data_type}")

                for output in outputs:
                    try:
                        generated_text = output.outputs[0].text
                        parsed_output = json.loads(generated_text)
                        
                        output_data = {"topic": parsed_output['topic']}
                        if data_type == 'article':
                            output_data["article_id"] = unique_id
                        else:
                            output_data["comment_id"] = unique_id
                    
                        validated_output = output_schema(**output_data)
                        print(json.dumps(validated_output.model_dump(), indent=4))

                        f.write(json.dumps(validated_output.model_dump()) + "\n")

                    except Exception as e:
                        print(f"Error processing {data_type} ID {unique_id}: {e}")
                        continue

    def classify_articles(self, file_paths: List[str], output_schema: BaseModel):
        self.classify(file_paths, output_schema, data_type='article')

    def classify_comments(self, file_paths: List[str], output_schema: BaseModel):
        self.classify(file_paths, output_schema, data_type='comment')

file_paths = ["/data/gpfs/projects/punim0478/guida/topic_stance_inferences/data/mock_analysis.csv"]

classifier = TopicClassifier(
    model_name="unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    prompt=PROMPT_TOPIC,
    article_output_file="/data/gpfs/projects/punim0478/guida/topic_stance_inferences/output/topic_articles.jsonl",  
    comment_output_file="/data/gpfs/projects/punim0478/guida/topic_stance_inferences/output/topic_comments.jsonl",
    cache_dir="/data/gpfs/projects/punim0478/guida"
)

classifier.classify_articles(file_paths, Output)
classifier.classify_comments(file_paths, Output)

torch.cuda.empty_cache()
