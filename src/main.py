import os 
import logging
import logging.config
import yaml

import pandas as pd 

from LM import LLM,JudgeLLM
from utils import (
    fill_out_prompt,
    dump_in_jsonl
)

from dotenv import load_dotenv
load_dotenv()

os.environ['GOOGLE_PROJECT_ID'] = os.getenv("GOOGLE_PROJECT_ID")
os.environ['GOOGLE_REGION'] = os.getenv("GOOGLE_REGION")
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "../vertex_ai_use_cred.json"

os.environ['LANGSMITH_API_KEY'] = os.getenv("LANGSMITH_API_KEY")
os.environ['LANGSMITH_PROJECT'] = os.getenv("LANGSMITH_PROJECT")
os.environ['LANGSMITH_TRACING_V2'] = "true"

with open("models.yaml","r") as f:
    models = yaml.safe_load(f)

test_models = models['TEST_MODELS']
judge_models = models['JUDGE_MODELS']
#-------------------------------------------------------------------------
logger = logging.getLogger(__name__)
#-------------------------------------------------------------------------
LIMIT_SAMPLE_SIZE = 2

df = pd.read_csv("../data/examples.csv")
df = df[:LIMIT_SAMPLE_SIZE]

evaluation_prompts_file_path = "../data/evaluation_prompts.csv"



test_responses = []
judge_responses = []

for index,row in df.iterrows():
    logger.info(f"Processing sample {index}")

    # STEP 1 : TEST RESPONSE 
    for test_model in test_models:
        model_instance = LLM(test_model)
        test_model_output  = model_instance.call_llm(row['full_prompt'],"test",sample_id = index)
        test_responses.append(test_model_output)
        dump_in_jsonl(test_model_output,"test_responses.jsonl")

        # STEP 2 : JUDGE RESPONSE 
        for j in judge_models:
            judge_model = JudgeLLM(j)
            judge_template = judge_model.fetch_evaluation_prompt(evaluation_prompts_file_path)

            # FILL OUT JUDGE TEMPLATE    
            filled_template = fill_out_prompt(
            judge_template,
            user_request = row['user_request'],
            context_document = row['context_document'],
            response = test_model_output['test model response']
            )

            # CALL JUDGE MODEL 
            judge_response = judge_model.call_llm(
            filled_template,
            intention = "judge",
            judgeded_model = test_model,
            sample_id = index
         )
            verdict = judge_model.get_verdict(judge_response['judge model response'])
            judge_response['verdict'] = verdict
            judge_responses.append(judge_response)
            dump_in_jsonl(judge_response,"judge_responses.jsonl")
