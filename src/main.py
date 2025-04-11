
import logging
import yaml
from typing import List,Dict,Any,Tuple
import pandas as pd 

from pipeline import LLM,JudgeLLM
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
    """Process a single sample through test models and judge models.
    
    This function handles the complete evaluation pipeline for a single sample:
    1. Runs the sample through each test model to generate responses
    2. Evaluates each test model response using the judge models
    
    Args:
        sample_id (int): Unique identifier for the sample being processed
        record (Dict[str, Any]): Dictionary containing sample data including 'full_prompt',
                                'user_request', and 'context_document'
        test_models (List[str]): List of test model identifiers to evaluate
        judge_models (List[str]): List of judge model identifiers to use for evaluation
        evaluation_prompts_file_path (str): Path to the CSV file containing evaluation prompts
        tracer_test_model (str, optional): Name of the LangSmith tracing project for test models
        tracer_judge_model (str, optional): Name of the LangSmith tracing project for judge models
        
    Returns:
        Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]: A tuple containing:
            - List of test model responses with metadata
            - List of judge evaluations for each test model response
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
            user_request = record['user_request'],
            context_document= record['context_document'],
            test_model_response= test_model_output['test model response'],
            skip_eval= test_model_output['Encountered Problems'],
            evaluation_prompt_file_path=evaluation_prompts_file_path,
            test_model= test_model,
            sample_id=sample_id,
            tracer_project = tracer_judge_model
        )

        judges_responses.append(judges_data)

    return test_responses,judges_responses




def main():
    """Main function to orchestrate the model evaluation process.
    
    This function:
    1. Loads environment variables and configurations
    2. Reads test and judge model specifications from a YAML config file
    3. Loads sample data from a CSV file
    4. Processes each sample by:
       - Generating responses from test models
       - Evaluating those responses using judge models
    5. Collects and logs all results
    
    The function uses configuration constants for:
    - Model configuration file path
    - Data file path
    - Sample range to process
    - Evaluation prompts file path
    
    Results are saved to JSONL files during processing for persistence.
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
    main()



