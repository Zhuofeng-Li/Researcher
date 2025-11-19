# ReviewRL-SFT

## Installation

```bash
uv venv
uv pip install -e "."
```

## Eval

You can find more commands in each file.

### Review generation 

```python
python evaluate/benchmark_deepreviewer.py --model-name ZhuofengLi/Qwen3-4B-Instruct-2507-DeepReview-lora-sft --num-samples -1
```
### Rule-based evaluation 
```python 
python evaluate/rule_evaluate.py evaluate/review/deepreviewer_Qwen3-4B-Instruct-2507-DeepReview-lora-sft_2025-11-14_17-12-41.json --modes fast
```

### RM-based evaluation 
```python 
python evaluate/rm_evaluate.py evaluate/review/deepreviewer_Qwen3-4B-Instruct-2507-DeepReview-lora-sft_2025-11-14_17-12-41.json --model_name gpt-4o-mini --max_workers 32
```
