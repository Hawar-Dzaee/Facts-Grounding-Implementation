
import logging
import yaml
from typing import List,Dict,Any,Tuple
import pandas as pd 
import time
from llm import LLM
from judge_eval import JudgeLLM
from utils import load_environment,dump_in_jsonl



logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
#-------------------------------------------

def process_sample(
        sample_id : int,
        record:Dict[str,Any],
        test_models:List[str],
        judge_models:List[str],
        evaluation_prompts_file_path:str,
        tracer_test_model:str=None,
        tracer_judge_model:str=None,
)->Tuple[List[Dict[str,Any]],List[Dict[str,Any]]]:
    """
    Process a single sample through the evaluation pipeline.
    
    Args:
        sample_id (int): Unique identifier for the sample
        record (Dict[str,Any]): Dictionary containing the sample data including full_prompt, user_request, and context_document
        test_models (List[str]): List of test model names to evaluate
        judge_models (List[str]): List of judge model names to use for evaluation
        evaluation_prompts_file_path (str): Path to the file containing evaluation prompts
        tracer_test_model (str, optional): Project name for test model tracing. Defaults to None.
        tracer_judge_model (str, optional): Project name for judge model tracing. Defaults to None.
    
    Returns:
        Tuple[List[Dict[str,Any]],List[Dict[str,Any]]]: A tuple containing:
            - List of test model responses
            - List of judge model evaluations
    """
    
    test_responses = []
    judges_responses = []

    # STEP 1 : TEST RESPONSE
    for test_model in test_models:
        model_instance = LLM(test_model)
        test_model_output  = model_instance.call_llm(
            record['full_prompt'],
            "test",
            sample_id = sample_id,
            tracer_project = tracer_test_model
            )
        test_responses.append(test_model_output)
        dump_in_jsonl(test_model_output,"test_responses.jsonl")


        # STEP 2 : JUDGE RESPONSE 
        judges = JudgeLLM(judge_models)
        judges_data = judges.calling_judges(
            sample_id = sample_id,
            user_request = record['user_request'],
            context_document= record['context_document'],
            test_model_response= test_model_output['test_model_response'],
            skip_eval= test_model_output['Encountered_Problems'],
            evaluation_prompt_file_path=evaluation_prompts_file_path,
            test_model= test_model,
            tracer_project = tracer_judge_model
        )

        judges_responses.append(judges_data)

    return test_responses,judges_responses




def main():
    """
    Main function to run the evaluation pipeline.
    
    This function:
    1. Loads environment variables
    2. Reads configuration from models.yaml
    3. Loads sample data from examples.csv
    4. Processes each sample through test and judge models
    5. Collects and logs results
    
    The function uses a sample size range defined by SAMPLE_SIZE_START and SAMPLE_SIZE_END
    to control the number of samples processed.
    """

    load_environment()

    CONFIG_PATH = "models.yaml"
    DATA_PATH = "../data/examples.csv"
    SAMPLE_SIZE_START = 0
    SAMPLE_SIZE_END = 2
    EVALUATION_PROMPTS_PATH = "../data/evaluation_prompts.csv"

    with open(CONFIG_PATH,"r") as f:
        models = yaml.safe_load(f)

        # Load models
    test_models = models['TEST_MODELS']
    judge_models = models['JUDGE_MODELS']


    # Load data
    try:
        df = pd.read_csv(DATA_PATH)[SAMPLE_SIZE_START:SAMPLE_SIZE_END]        
        records = df.to_dict(orient='records')
        logger.info(f"Loaded {len(records)} samples for evaluation")
    except Exception as e:
        logger.error(f"Error loading data from {DATA_PATH}: {e}")
        return
    
    # Process samples
    all_test_responses = []
    all_judge_responses = []

    for index, record in enumerate(records):
        logger.info(f"Processing sample {index}")
        test_responses,judges_responses = process_sample(
        sample_id = index,
            record = record,
            test_models = test_models,
            judge_models = judge_models,
            evaluation_prompts_file_path = EVALUATION_PROMPTS_PATH,
            tracer_test_model = "test_model",
            tracer_judge_model = "judge_model"
        )

        all_test_responses.extend(test_responses)
        all_judge_responses.extend(judges_responses)
    
    logger.info(f"Completed processing {len(records)} samples")


if __name__ == "__main__":
    start_time = time.perf_counter()
    main()
    end_time = time.perf_counter()
    logger.info(f"Total time taken: {end_time - start_time} seconds")



