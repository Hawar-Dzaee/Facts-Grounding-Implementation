import logging
import json
from typing import List
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


from langchain.chat_models import init_chat_model
from langchain.callbacks.tracers import LangChainTracer


from utils import dump_in_jsonl


class LLM:
    def __init__(self,model:str):
        self.model = model

    def call_llm(self,text:str,intention:str,temperature=0.0,tracer_project:str=None,**kwargs):
        logger.debug(f"Calling : {self.model} | Intention : {intention} ...") 
        try:
            response = init_chat_model(model=self.model,temperature=temperature)
            response = response.invoke(
                text,
                config= {"callbacks": [LangChainTracer(project_name=tracer_project)]} if tracer_project else {}
                ).content
            logger.info(f"Successfully called : {self.model} | Intention : {intention}.")
            result = {
                    f'{intention} model':self.model,
                    f'Encountered Problems':False,
                    f'{intention} model response':response
                }
        except Exception as e:  
            logger.error(f"Failed to call : {self.model} | Intention : {intention} | Error : {e}")
            result = {
                    f'{intention} model':self.model,
                    f'Encountered Problems':True,
                    f'{intention} model response':str(e)
                }
            
        kwargs.update(result)
        return kwargs

    

class JudgeLLM(LLM):
    def __init__(self,judges:List[str]):
        self.judges = [LLM(i) for i in judges]

    
    def calling_judges(self,
                       user_request:str,
                       context_document:str,
                       test_model_response:str,
                       test_model :str,
                       test_model_response_available:bool,
                       sample_id : str,
                       evaluation_prompt_file_path:str,
                       tracer_project:str=None
                       ):
        
        judge_responses =  []
        if test_model_response_available:
            for j in self.judges:
                judge_response = {
                    'judge model':j.model,
                    'sample id':sample_id,
                    'judge model':j.model,
                    "Encountered Problems": "Not applicable",
                    'judge model response':"Not applicable",
                    'verdict':"Not applicable"
                }

        else : 
            for j in self.judges:

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
                    judge_response["judge model response"],
                    evaluation_method
                    )
                
                judge_response['verdict'] = verdict

        judge_responses.append(judge_response)
        dump_in_jsonl(judge_response,"judge_responses.jsonl")

        return judge_responses
            


    def fetch_evaluation_prompt(self,evaluation_prompt_file_path:str,evaluation_method):
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
                if "Final Answer" in raw_output: 
                    verdict = raw_output.split("Final Answer: ")[1]
            logger.info(f"{verdict}")
            return verdict
        
        except Exception as e:
            logger.error(f"Error : {e}")
            verdict = None





