import logging
import json
from typing import List
import pandas as pd
import re
import concurrent.futures
from llm import LLM
from utils import dump_in_jsonl


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



class JudgeLLM:
    """
    A class that manages multiple judge LLMs for evaluating model responses.
    
    This class handles the orchestration of multiple judge models to evaluate the accuracy
    of responses from test models against given context and user requests.
    
    Attributes:
        judges (List[str]): List of judge model names to be used for evaluation
    """

    def __init__(self,judges:List[str]):
        """
        Initialize the JudgeLLM with a list of judge models.
        
        Args:
            judges (List[str]): List of judge model names to be used for evaluation
        """
        self.judges = judges

    
    def calling_judge(self,
                       judge_model : str,
                       user_request:str,
                       context_document:str,
                       test_model_response:str,
                       test_model :str,
                       skip_eval:bool,
                       sample_id : str,
                       evaluation_prompt_file_path:str,
                       tracer_project:str=None
                       ):
        """
        Call a single judge model to evaluate a test model's response.
        
        Args:
            judge_model (str): Name of the judge model to use
            user_request (str): The original user request/query
            context_document (str): The context document used for evaluation
            test_model_response (str): The response from the test model to evaluate
            test_model (str): Name of the test model being evaluated
            skip_eval (bool): Whether to skip the evaluation
            sample_id (str): Unique identifier for the evaluation sample
            evaluation_prompt_file_path (str): Path to the file containing evaluation prompts
            tracer_project (str, optional): Project name for tracing purposes
            
        Returns:
            dict: Dictionary containing the judge's response and evaluation verdict
        """
        j = LLM(judge_model)

        if skip_eval:
            judge_response = {
                'sample_id':sample_id,
                'judge_model':j.model,
                'judged_model':test_model,
                "Encountered_Problems": "Not applicable",
                'judge_model_response':"Not applicable",
                'verdict':"Not applicable"
            }
            dump_in_jsonl(judge_response,"judge_responses.jsonl")

        else : 
            if "anthropic" in j.model:
                evaluation_method = "implicit_span_level"
            else : 
                evaluation_method = "json"

            judge_template = self.fetch_evaluation_prompt(evaluation_prompt_file_path,evaluation_method)

            filled_prompt = self.fill_out_prompt(
                judge_template,
                user_request = user_request,
                context_document = context_document,
                response = test_model_response
            )

            judge_response = j.call_llm(
                filled_prompt,
                intention= "judge",
                judged_model =  test_model,
                sample_id = sample_id,
                tracer_project = tracer_project
            )
            
            verdict = self.get_verdict(
                judge_response["judge_model_response"],
                evaluation_method
                )
            
            judge_response['verdict'] = verdict
            dump_in_jsonl(judge_response,"judge_responses.jsonl")

        return judge_response
    
    
    def calling_judges(self,
                        user_request:str,
                        context_document:str,
                        test_model_response:str,
                        test_model :str,
                        skip_eval:bool,
                        sample_id : str,
                        evaluation_prompt_file_path:str,
                        tracer_project:str=None):
        """
        Call multiple judge models in parallel to evaluate a test model's response.
        
        Args:
            user_request (str): The original user request/query
            context_document (str): The context document used for evaluation
            test_model_response (str): The response from the test model to evaluate
            test_model (str): Name of the test model being evaluated
            skip_eval (bool): Whether to skip the evaluation
            sample_id (str): Unique identifier for the evaluation sample
            evaluation_prompt_file_path (str): Path to the file containing evaluation prompts
            tracer_project (str, optional): Project name for tracing purposes
            
        Returns:
            List[dict]: List of dictionaries containing each judge's response and evaluation verdict
        """
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.calling_judge,
                                        j,
                                        user_request,
                                        context_document,
                                        test_model_response,
                                        test_model,
                                        skip_eval,sample_id,evaluation_prompt_file_path,tracer_project) for j in self.judges]
            results = list(concurrent.futures.as_completed(futures))
            results = [result.result() for result in results]
            return results

    def fetch_evaluation_prompt(self,evaluation_prompt_file_path:str,evaluation_method):
        """
        Fetch the evaluation prompt based on the specified evaluation method.
        
        Args:
            evaluation_prompt_file_path: Path to the CSV file containing evaluation prompts
            evaluation_method: The evaluation method to use (e.g., 'json', 'implicit_span_level')
            
        Returns:
            The evaluation prompt template string for the specified method
        """
        evaluation_prompt_file = pd.read_csv(evaluation_prompt_file_path)
        evaluation_prompt = evaluation_prompt_file.loc[evaluation_prompt_file["evaluation_method"] == evaluation_method, "evaluation_prompt"].values[0]
        return evaluation_prompt

    def fill_out_prompt(self,prompt_template: str, **kwargs) -> str:
        """
        Format a prompt template with {{variable}} style placeholders.
        
        Args:
            prompt_template: Template string with {{variable}} placeholders
            **kwargs: Variables to substitute into the template
            
        Returns:
            Formatted prompt string
        """
        try:
            for key, value in kwargs.items():
                placeholder = f"{{{{{key}}}}}"
                prompt_template = prompt_template.replace(placeholder, str(value))
            return prompt_template
        except Exception as e:
            logger.error(f"Error formatting prompt template: {e}")
            raise


    def get_verdict(self,raw_output:str,evaluation_method):
        """
        Extract the verdict from the judge model's raw output based on the evaluation method.
        
        Args:
            raw_output: The raw text response from the judge model
            evaluation_method: The evaluation method used (e.g., 'json', 'implicit_span_level')
            
        Returns:
            A verdict string (e.g., 'Accurate', 'Inaccurate') or None if parsing fails
        """
        try : 
            if evaluation_method == 'json':
                processed_output = raw_output.strip("```json").strip()
                breakdown_list = [json.loads(line) for line in processed_output.strip().split("\n")]
                labels = [i['label'] for i in breakdown_list]

                if not any(label in ["unsupported","contradictory"] for label in labels):
                    verdict = "Accurate"
                else : 
                    verdict = "Inaccurate"
            else :
                match = re.search(r'Final Answer:\s*(Accurate|Inaccurate)', raw_output)
                if match:
                    verdict = match.group(1)  # Return just the "Accurate" or "Inaccurate" part
                else:
                    verdict = None

            logger.info(f"{verdict}")
            return verdict
        
        except Exception as e:
            logger.error(f"Error in get_verdict : {e}")
            verdict = None