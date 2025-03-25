
import logging
import json
import pandas as pd
# from google.generativeai import GenerativeModel as genai

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


from langchain.chat_models import init_chat_model



class LM:

    def __init__(self,model:str):
        self.model = model
        assert ":" in model, "All models must be in format 'provider:version'"
        self.model_provider,self.model_version = self.model.split(":")

        

    def call_lm(self,text:str,intention:str,temperature=0.0,**kwargs):
        logger.info(f"Calling : {self.model} | Intention : {intention} ...") 
        try:
            response = init_chat_model(model=self.model,temperature=temperature)
            response = response.invoke(text)
            response = response.content
            logger.info(f"Successfully called : {self.model} | Intention : {intention}.")

            result = {
                    f'{intention} model provider':self.model_provider,
                    f'{intention} model version':self.model_version,
                    f'Encountered Problems':False,
                    f'{intention} model response':response
                }
        except Exception as e:  
            logger.error(f"Failed to call : {self.model} | Intention : {intention} | Error : {e}")
            result = {
                    f'{intention} model provider':self.model_provider,
                    f'{intention} model version':self.model_version,
                    f'Encountered Problems':True,
                    f'{intention} model input token count':None,
                    f'{intention} model output token count':None,
                    f'{intention} model response':str(e)
                }
            
        result.update(kwargs)
        return result


    

class JudgeLM(LM):
    def __init__(self,model:str,evaluation_method:str=None):
        super().__init__(model)
        self.evaluation_method = evaluation_method

        if self.evaluation_method is None:
            self.evaluation_method = "implicit_span_level"  if self.model_provider == "anthropic" else 'json'


    def fetch_evaluation_prompt(self,evaluation_prompt_file_path:str):
        evaluation_prompt_file = pd.read_csv(evaluation_prompt_file_path)
        evaluation_prompt = evaluation_prompt_file.loc[evaluation_prompt_file["evaluation_method"] == self.evaluation_method, "evaluation_prompt"].values[0]
        return evaluation_prompt
    

    def get_verdict(self,raw_output:str):
        logger.info(f"Getting verdict for : {self.model} | Evaluation Method : {self.evaluation_method}...")

        try : 
            if self.evaluation_method == 'json':
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
            logger.info(f"Successfully got verdict for : {self.model} | Evaluation Method : {self.evaluation_method}.")
            return verdict
        
        except Exception as e:
            logger.error(f"Failed to get verdict for : {self.model} due to LLMs unexpected output format| Evaluation Method : {self.evaluation_method} | Error : {e}")
            verdict = None





