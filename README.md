# Facts Grounding implementation

## Overview
This repository implements a framework for evaluating Language Models (LLMs) using judge models. The system consists of two main components:
1. **Test LLMs** - These models generate responses based on user queries.
2. **Judge LLMs** - These models evaluate the accuracy and quality of test model responses.

This implementation is based on the paper ["Facts-Grounding: A New Benchmark for Evaluating the Factuality of Large Language Models"](https://arxiv.org/abs/2501.03200). You can also refer to the related [DeepMind blog post](https://deepmind.google/discover/blog/facts-grounding-a-new-benchmark-for-evaluating-the-factuality-of-large-language-models/) for a more detailed discussion on the methodology and findings.

## Features
- Calls test models to generate responses for user queries.
- Uses judge models to assess the test model responses against provided context documents.
- Supports multiple judge models for a more reliable evaluation.
- Logs all interactions for debugging and performance monitoring.
- Saves results in JSONL format for easy analysis.

## Files and Structure
```
.
├── LM.py                 # Contains LLM and JudgeLLM classes
├── main.py               # Main execution script
├── models.yaml           # Configuration file for specifying test and judge models
├── utils.py              # Utility functions (loading environment, dumping JSONL, etc.)
├── data/
│   ├── examples.csv      # Sample dataset containing prompts and context documents
│   ├── evaluation_prompts.csv # Predefined evaluation prompts for judge models
└── README.md             # Documentation (this file)
```

## Setup Instructions
### 1. Install Dependencies
Ensure you have Python installed. Then, install required dependencies:
```bash
pip install -r requirements.txt
```

### 2. Set Up Environment
Make sure to define environment variables or configurations required for model initialization. This can be done by modifying `utils.py` and `models.yaml`.

### 3. Run the Script
To execute the evaluation pipeline:
```bash
python main.py
```

## Code Explanation
### `LM.py`
- **LLM Class**: Calls a specified LLM model and retrieves responses.
- **JudgeLLM Class**: Uses a list of judge models to evaluate test model outputs.

### `main.py`
- Loads test and judge models from `models.yaml`.
- Iterates through sample prompts in `examples.csv`.
- Calls test models to generate responses.
- Uses judge models to evaluate the responses.
- Logs and stores the results.

## Output
- `test_responses.jsonl`: Stores responses from test models.
- `judge_responses.jsonl`: Stores evaluations from judge models.

## Customization
- Modify `models.yaml` to change test and judge models.
- Update `examples.csv` with new sample queries.
- Adjust `evaluation_prompts.csv` for different evaluation criteria.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.