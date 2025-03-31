import pytest 
import os
from src.utils import fill_out_prompt,dump_in_jsonl
import json




def test_fill_out_prompt():
    template = """My name is {{name}}, I am {{age}} years old.
    `This is not a placeholder` **neither is this** 
    {"sentence" : "Even though it is in a dictionary it is not a placeholder"}"""

    result = fill_out_prompt(template,name="Hawar",age=30)

    assert result == """My name is Hawar, I am 30 years old.
    `This is not a placeholder` **neither is this** 
    {"sentence" : "Even though it is in a dictionary it is not a placeholder"}"""




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









