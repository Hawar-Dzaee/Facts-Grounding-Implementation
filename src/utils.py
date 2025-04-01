import os 
import logging
import json 
from typing import Dict,Any
from pathlib import Path

from dotenv import load_dotenv

logger = logging.getLogger(__name__)



def load_environment():
    """Load environment variables and configure settings."""
    load_dotenv()
    
    # Set required environment variables
    os.environ['GOOGLE_PROJECT_ID'] = os.getenv("GOOGLE_PROJECT_ID", "")
    os.environ['GOOGLE_REGION'] = os.getenv("GOOGLE_REGION", "")
    
    # Use pathlib for more robust path handling
    creds_path = Path("../vertex_ai_use_cred.json").resolve()
    if not creds_path.exists():
        logger.warning(f"Credentials file not found at {creds_path}")
    
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = str(creds_path)
    
    # LangSmith configuration
    os.environ['LANGSMITH_API_KEY'] = os.getenv("LANGSMITH_API_KEY", "")
    os.environ['LANGSMITH_PROJECT'] = os.getenv("LANGSMITH_PROJECT", "")
    os.environ['LANGSMITH_TRACING_V2'] = "true"


def dump_in_jsonl(data: Dict[str, Any], file_name: str) -> None:
    """
    Append data as a JSON line to the specified file.
    
    Args:
        data: Dictionary to write as JSON
        file_name: Path to the output file
    """
    file_path = Path(file_name)
    
    try:
        # Create directory if it doesn't exist
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'a') as file:
            json.dump(data, file)
            file.write("\n")
    except Exception as e:
        logger.error(f"Error writing to {file_name}: {e}")