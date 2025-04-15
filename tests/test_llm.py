import pytest
from unittest.mock import patch, MagicMock
from src.llm import LLM

@pytest.fixture
def llm():
    return LLM(model="test-model")

def test_llm_initialization():
    model_name = "test-model"
    llm = LLM(model=model_name)
    assert llm.model == model_name

@patch('src.llm.init_chat_model')
def test_call_llm_success(mock_init_chat_model, llm):
    # Setup mock
    mock_chat_model = MagicMock()
    mock_chat_model.invoke.return_value = MagicMock(content="Test response")
    mock_init_chat_model.return_value = mock_chat_model

    # Test parameters
    text = "Test input"
    intention = "test_intention"
    temperature = 0.5
    tracer_project = "test-project"
    additional_kwargs = {"key": "value"}

    # Call the method
    result = llm.call_llm(
        text=text,
        intention=intention,
        temperature=temperature,
        tracer_project=tracer_project,
        **additional_kwargs
    )

    # Assertions
    assert result[f"{intention}_model"] == llm.model
    assert not result["Encountered_Problems"]
    assert result[f"{intention}_model_response"] == "Test response"
    assert result["key"] == "value"
    
    # Verify mock was called correctly
    mock_init_chat_model.assert_called_once_with(
        model=llm.model,
        temperature=temperature
    )
    mock_chat_model.invoke.assert_called_once()

@patch('src.llm.init_chat_model')
def test_call_llm_error(mock_init_chat_model, llm):
    # Setup mock to raise an exception
    mock_init_chat_model.side_effect = Exception("Test error")

    # Test parameters
    text = "Test input"
    intention = "test_intention"

    # Call the method
    result = llm.call_llm(
        text=text,
        intention=intention
    )

    # Assertions
    assert result[f"{intention}_model"] == llm.model
    assert result["Encountered_Problems"]
    assert "Test error" in result[f"{intention}_model_response"]

@patch('src.llm.init_chat_model')
def test_call_llm_without_tracer(mock_init_chat_model, llm):
    # Setup mock
    mock_chat_model = MagicMock()
    mock_chat_model.invoke.return_value = MagicMock(content="Test response")
    mock_init_chat_model.return_value = mock_chat_model

    # Call without tracer_project
    result = llm.call_llm(
        text="Test input",
        intention="test_intention"
    )

    # Assertions
    assert not result["Encountered_Problems"]
    mock_chat_model.invoke.assert_called_once()
