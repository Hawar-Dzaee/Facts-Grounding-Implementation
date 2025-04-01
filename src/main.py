
import logging
import yaml
from typing import List

import pandas as pd 

from LM import LLM,JudgeLLM
from utils import load_environment,dump_in_jsonl


load_environment()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)



with open("models.yaml","r") as f:
    models = yaml.safe_load(f)

test_models = models['TEST_MODELS']
judge_models = models['JUDGE_MODELS']
#-------------------------------------------
LIMIT_SAMPLE_SIZE = 1

df = pd.read_csv("../data/examples.csv")
df = df[:LIMIT_SAMPLE_SIZE]

evaluation_prompts_file_path = "../data/evaluation_prompts.csv"



test_responses = []
judges_repsonses = []

for index,row in df.iterrows():
    logger.info(f"Processing sample {index}")

    # STEP 1 : TEST RESPONSE 
    for test_model in test_models:
        model_instance = LLM(test_model)
        test_model_output  = model_instance.call_llm(row['full_prompt'],"test",sample_id = index)
        test_responses.append(test_model_output)
        dump_in_jsonl(test_model_output,"test_responses.jsonl")

        # STEP 2 : JUDGE RESPONSE 
        judges = JudgeLLM(judge_models)
        judges_data = judges.calling_judges(
            user_request = row['user_request'],
            context_document= row['context_document'],
            test_model_response= test_model_output['test model response'],
            test_model= test_model,
            sample_id=index,
            evaluation_prompt_file_path=evaluation_prompts_file_path
        )



