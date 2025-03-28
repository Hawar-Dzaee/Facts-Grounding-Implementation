import os 
import logging
import logging.config
import yaml

import pandas as pd 

from LM import LM,JudgeLM
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



#-------------------------------------------------------------------------
logger = logging.getLogger(__name__)
#-------------------------------------------------------------------------
LIMIT_SAMPLE_SIZE = 2

df = pd.read_csv("../data/examples.csv")
df = df[:LIMIT_SAMPLE_SIZE]

evaluation_prompts_file_path = "../data/evaluation_prompts.csv"


test_models = [
    "openai:gpt-3.5-turbo",
    "openai:gpt-4o",
    # "anthropic:claude-3-5-sonnet-20240620",
   #  "google:gemini-1.5-pro",
    ]

judge_models = [
    # "openai:gpt-4o",
    "anthropic:claude-3-5-sonnet-20240620",
    "google_vertexai:gemini-1.5-pro",
    ]



test_responses = []
judge_responses = []

for index,row in df.iterrows():
    logger.info(f"Processing sample {index}")

    # STEP 1 : TEST RESPONSE 
    for test_model in test_models:
        model_instance = LM(test_model)
        test_model_output  = model_instance.call_lm(row['full_prompt'],"test",sample_id = index)
        test_responses.append(test_model_output)
        dump_in_jsonl(test_model_output,"test_responses.jsonl")

        # STEP 2 : JUDGE RESPONSE 
        for j in judge_models:
            judge_model = JudgeLM(j)
            judge_template = judge_model.fetch_evaluation_prompt(evaluation_prompts_file_path)

            # FILL OUT JUDGE TEMPLATE    
            filled_template = fill_out_prompt(
            judge_template,
            user_request = row['user_request'],
            context_document = row['context_document'],
            response = test_model_output['test model response']
            )

            # CALL JUDGE MODEL 
            judge_response = judge_model.call_lm(
            filled_template,
            intention = "judge",
            judgeded_model = test_model,
            sample_id = index
         )
            verdict = judge_model.get_verdict(judge_response['judge model response'])
            judge_response['verdict'] = verdict
            judge_responses.append(judge_response)
            dump_in_jsonl(judge_response,"judge_responses.jsonl")
