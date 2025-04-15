import logging

from langchain.chat_models import init_chat_model
from langchain.callbacks.tracers import LangChainTracer



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLM:
    """
    A class to handle interactions with Language Learning Models (LLMs).
    
    This class provides functionality to initialize and call LLM models with various configurations.
    It includes error handling and logging capabilities.
    
    Attributes:
        model (str): The name or identifier of the LLM model to be used.
    """

    def __init__(self, model: str):
        """
        Initialize the LLM class with a specific model.
        
        Args:
            model (str): The name or identifier of the LLM model to be used.
        """
        self.model = model

    def call_llm(self, text: str, intention: str, temperature: float = 0.0, tracer_project: str = None, **kwargs):
        """
        Call the LLM model with the provided text and parameters.
        
        Args:
            text (str): The input text to be processed by the LLM.
            intention (str): The purpose or intention of the LLM call (used for logging and result organization).
            temperature (float, optional): Controls randomness in the model's output. Defaults to 0.0.
            tracer_project (str, optional): Name of the project for LangChain tracing. Defaults to None.
            **kwargs: Additional keyword arguments to be included in the result.
            
        Returns:
            dict: A dictionary containing:
                - Model information
                - Success/failure status
                - Model response or error message
                - Any additional kwargs provided
        """
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
    



    







