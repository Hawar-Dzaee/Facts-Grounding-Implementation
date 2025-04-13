import logging

from langchain.chat_models import init_chat_model
from langchain.callbacks.tracers import LangChainTracer

from utils import dump_in_jsonl


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
                    f'{intention}_model':self.model,
                    f'Encountered_Problems':False,
                    f'{intention}_model_response':response
                }
        except Exception as e:  
            logger.error(f"Failed to call : {self.model} | Intention : {intention} | Error : {e}")
            result = {
                    f'{intention}_model':self.model,
                    f'Encountered_Problems':True,
                    f'{intention}_model_response':str(e)
                }
            
        kwargs.update(result)
        return kwargs
    



    







