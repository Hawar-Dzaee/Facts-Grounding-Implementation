import pytest 
import os
import json
from pathlib import Path
from unittest.mock import patch, mock_open, MagicMock


from src.utils import dump_in_jsonl,load_environment





def test_load_environment_with_file_exists():
    """Test load_environment when the credentials file exists."""
    with patch('src.utils.load_dotenv'), \
         patch('src.utils.Path.exists', return_value=True), \
         patch('src.utils.Path.resolve', return_value=Path('../vertex_ai_use_cred.json')), \
         patch.dict(os.environ, {}, clear=True):
        
        # Call the function
        load_environment()
        
        # Verify environment variables were set correctly
        assert os.environ.get('GOOGLE_PROJECT_ID') == ''
        assert os.environ.get('GOOGLE_REGION') == ''
        assert os.environ.get('GOOGLE_APPLICATION_CREDENTIALS') == '../vertex_ai_use_cred.json'
        assert os.environ.get('LANGSMITH_API_KEY') == ''
        assert os.environ.get('LANGSMITH_TRACING_V2') == 'true'


def test_load_environment_file_not_exists():
    """Test load_environment when the credentials file does not exist."""
    with patch('src.utils.load_dotenv'), \
         patch('src.utils.Path.exists', return_value=False), \
         patch('src.utils.Path.resolve', return_value=Path('../vertex_ai_use_cred.json')), \
         patch('src.utils.logger.warning') as mock_warning, \
         patch.dict(os.environ, {}, clear=True):
        
        # Call the function
        load_environment()
        
        # Verify warning was logged
        mock_warning.assert_called_once_with(
            "Credentials file not found at ../vertex_ai_use_cred.json"
        )
        
        # Verify environment variables were still set
        assert os.environ.get('GOOGLE_APPLICATION_CREDENTIALS') == '../vertex_ai_use_cred.json'


#------------------------------------------------
@pytest.fixture
def sample_data():
    """Fixture providing sample JSON data."""
    return {"name": "Hawar", "age": 30}

def test_dump_in_jsonl_valid(tmp_path, sample_data):
    """Test that data is correctly written to a JSONL file."""
    file_path = tmp_path / "test.jsonl"

    dump_in_jsonl(sample_data, str(file_path))

    # Verify file exists
    assert file_path.exists()

    # Verify content is correctly written
    with open(file_path, "r") as f:
        lines = f.readlines()
    
    assert len(lines) == 1  # Only one JSON entry should be written
    assert json.loads(lines[0]) == sample_data  # Ensure correct data


def test_dump_in_jsonl_appends_data(tmp_path, sample_data):
    """Test that dump_in_jsonl correctly appends data."""
    file_path = tmp_path / "test.jsonl"

    dump_in_jsonl(sample_data, str(file_path))
    dump_in_jsonl(sample_data, str(file_path))

    with open(file_path, "r") as f:
        lines = f.readlines()

    assert len(lines) == 2  # Should have two entries
    assert json.loads(lines[0]) == sample_data
    assert json.loads(lines[1]) == sample_data









