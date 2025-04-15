import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from src.llm import LLM
from src.judge_eval import JudgeLLM
import json
import tempfile
import shutil
from unittest.mock import patch

@pytest.fixture(scope="session")
def temp_dir():
    # Create a temporary directory for the session
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup after all tests are done
    shutil.rmtree(temp_dir)

@pytest.fixture
def output_file(temp_dir):
    # Create a temporary file path for each test
    return os.path.join(temp_dir, "judge_responses.jsonl")

@pytest.fixture
def sample_judges():
    return ["gpt-3.5-turbo", "claude-2"]

@pytest.fixture
def sample_data():
    return {
        "user_request": "What is the capital of France?",
        "context_document": "Paris is the capital of France.",
        "test_model_response": "The capital of France is Paris.",
        "test_model": "test-model-1",
        "sample_id": "test-1"
    }

@pytest.fixture
def evaluation_prompt_file(temp_dir):
    # Create a temporary CSV file with evaluation prompts
    temp_file = os.path.join(temp_dir, "evaluation_prompts.csv")
    with open(temp_file, 'w') as f:
        f.write("evaluation_method,evaluation_prompt\n")
        f.write("json,Evaluate if the response is accurate based on the context: {{user_request}}\nContext: {{context_document}}\nResponse: {{response}}\n")
        f.write("implicit_span_level,Evaluate the response: {{user_request}}\nContext: {{context_document}}\nResponse: {{response}}\n")
    return temp_file

def test_judge_llm_initialization(sample_judges):
    judge_llm = JudgeLLM(sample_judges)
    assert judge_llm.judges == sample_judges

def test_fetch_evaluation_prompt(evaluation_prompt_file):
    judge_llm = JudgeLLM(["test-judge"])
    prompt = judge_llm.fetch_evaluation_prompt(evaluation_prompt_file, "json")
    assert isinstance(prompt, str)
    assert "Evaluate if the response is accurate" in prompt

def test_fill_out_prompt():
    judge_llm = JudgeLLM(["test-judge"])
    template = "Question: {{user_request}}\nAnswer: {{response}}"
    filled = judge_llm.fill_out_prompt(template, user_request="test", response="test answer")
    assert "Question: test" in filled
    assert "Answer: test answer" in filled

def test_get_verdict_json():
    judge_llm = JudgeLLM(["test-judge"])
    json_output = """```json
    {"label": "supported"}
    {"label": "supported"}
    ```"""
    verdict = judge_llm.get_verdict(json_output, "json")
    assert verdict == "Accurate"

def test_get_verdict_implicit():
    judge_llm = JudgeLLM(["test-judge"])
    implicit_output = "Final Answer: Accurate"
    verdict = judge_llm.get_verdict(implicit_output, "implicit_span_level")
    assert verdict == "Accurate"

def test_calling_judge_skip_eval(sample_judges, sample_data, evaluation_prompt_file, output_file):
    with patch('src.judge_eval.dump_in_jsonl') as mock_dump:
        judge_llm = JudgeLLM(sample_judges)
        response = judge_llm.calling_judge(
            judge_model=sample_judges[0],
            user_request=sample_data["user_request"],
            context_document=sample_data["context_document"],
            test_model_response=sample_data["test_model_response"],
            test_model=sample_data["test_model"],
            skip_eval=True,
            sample_id=sample_data["sample_id"],
            evaluation_prompt_file_path=evaluation_prompt_file
        )
        
        assert response["sample_id"] == sample_data["sample_id"]
        assert response["judged_model"] == sample_data["test_model"]
        assert response["verdict"] == "Not applicable"
        
        # Verify that dump_in_jsonl was called with the correct data
        mock_dump.assert_called_once()
        call_args = mock_dump.call_args[0]
        assert call_args[0]["sample_id"] == sample_data["sample_id"]
        assert call_args[0]["judged_model"] == sample_data["test_model"]
        assert call_args[0]["verdict"] == "Not applicable"

def test_calling_judges(sample_judges, sample_data, evaluation_prompt_file, output_file):
    with patch('src.judge_eval.dump_in_jsonl') as mock_dump:
        judge_llm = JudgeLLM(sample_judges)
        responses = judge_llm.calling_judges(
            user_request=sample_data["user_request"],
            context_document=sample_data["context_document"],
            test_model_response=sample_data["test_model_response"],
            test_model=sample_data["test_model"],
            skip_eval=True,
            sample_id=sample_data["sample_id"],
            evaluation_prompt_file_path=evaluation_prompt_file
        )
        
        assert len(responses) == len(sample_judges)
        for response in responses:
            assert response["sample_id"] == sample_data["sample_id"]
            assert response["judged_model"] == sample_data["test_model"]
        
        # Verify that dump_in_jsonl was called the correct number of times
        assert mock_dump.call_count == len(sample_judges)
        for call in mock_dump.call_args_list:
            call_args = call[0]
            assert call_args[0]["sample_id"] == sample_data["sample_id"]
            assert call_args[0]["judged_model"] == sample_data["test_model"]
            assert call_args[0]["verdict"] == "Not applicable"
