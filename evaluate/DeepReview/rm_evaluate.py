
import json
import os
import random

import openai
from tqdm import tqdm
import argparse


# System prompt for the evaluator
SYSTEM_PROMPT = """
---***---
SYSTEM PROMPT 
---***---

You are a neutral arbitrator evaluating peer review comments for academic papers. Your role is to analyze and compare reviews through careful, evidence-based assessment. Your judgments must be strictly based on verifiable evidence from the paper and reviews.

For each evaluation, you must:

1. Thoroughly understand the paper by analyzing:
   - Research objectives and contributions
   - Methodology and experiments
   - Claims and evidence
   - Results and conclusions

2. For each review, methodically examine:
   - Claims made about the paper
   - Evidence cited to support claims
   - Technical assessments and critiques
   - Suggested improvements

3. Compare reviews systematically using:
   - Direct quotes from paper and reviews
   - Specific examples and counterexamples
   - Clear reasoning chains
   - Objective quality metrics

You will evaluate reviews based on these key aspects:

**Technical Accuracy**
- Are claims consistent with paper content?
- Is evidence properly interpreted?
- Are technical assessments valid?
- Are critiques well-supported?

**Constructive Value**
- How actionable is the feedback?
- Are suggestions specific and feasible?
- Is criticism balanced with strengths?
- Would authors understand how to improve?

**Analytical Depth**
- How thoroughly are key aspects examined?
- Is analysis appropriately detailed?
- Are important elements addressed?
- Is assessment comprehensive?

**Communication Clarity**
- Are points clearly articulated?
- Is feedback specific and concrete?
- Is reasoning well-explained?
- Are examples effectively used?

For each aspect and overall judgment, you must:
1. Provide specific evidence from source materials
2. Quote directly from paper and reviews
3. Explain your reasoning in detail
4. Consider alternative interpretations

**Input Format:**
- Complete paper text
- Assistant A's review 
- Assistant B's review

**Output Format:**

For each aspect:

```
**[Aspect Name] - Evidence Analysis:**
- From Assistant A:
  [Direct quotes and specific examples]
  [Detailed analysis of evidence]
- From Assistant B:
  [Direct quotes and specific examples]
  [Detailed analysis of evidence]
- Comparative Assessment:
  [Evidence-based comparison]
  [Clear reasoning chain]

**[Aspect Name] - Judgment:**
**Evidence-Based Reason:** [Detailed justification citing specific evidence]
**Better Assistant:** [A or B or Tie]
- If Tie: Explain why both reviews are equally strong on this aspect
```

Conclude with:

```
**Comprehensive Analysis:**
[Synthesis of evidence across aspects]
[Analysis of relative strengths]
[Discussion of key differences or similarities]

**Overall Judgment:**
**Evidence-Based Reason:** [Detailed justification synthesizing key evidence]
**Better Assistant:** [A or B or Tie]
- If Overall Tie: Explain why both reviews are comparable in overall quality
```

Key Requirements:
1. Base all judgments on concrete evidence
2. Quote directly from source materials
3. Provide detailed reasoning chains
4. Maintain neutral arbitrator perspective
5. Judge Tie when evidence shows equal strength
6. Always justify Tie decisions with specific evidence

When judging Tie:
- Ensure both reviews demonstrate similar levels of quality
- Provide explicit evidence showing comparable strengths
- Explain why differences are not significant enough to favor one over the other
- Consider both quantity and quality of evidence

Begin analysis after receiving complete materials. Take time to examine evidence thoroughly and provide detailed, justified assessments.
"""


class ReviewProcessor:
    """Handles the extraction and processing of reviews from different sources."""

    @staticmethod
    def extract_review_content(pred_context):
        """
        Extract the review content from the prediction context.

        Args:
            pred_context: Raw prediction data that contains the review

        Returns:
            str: Extracted review content
        """
        try:
            # First attempt to extract from boxed format
            return pred_context.split(r'\boxed_review{')[-1].split('\n}')[0]
        except:
            # Alternative extraction if the first method fails
            if isinstance(pred_context, dict) and 'output' in pred_context:
                return pred_context['output'].split(r'\boxed_review{')[-1].split('\n}')[0]
            else:
                # Return as is if extraction fails
                return pred_context


class DataManager:
    """Handles data loading, preparation and management."""

    @staticmethod
    def load_data(file_path):
        """
        Load data from a JSON file.

        Args:
            file_path: Path to the JSON file

        Returns:
            list: Loaded data
        """
        with open(file_path, encoding='utf-8') as f:
            return json.load(f)

    @staticmethod
    def prepare_comparison_data(dataA, dataB, args):
        """
        Prepare data for comparison between standard and fast mode reviews.

        Returns:
            list: Prepared data for comparison
        """
        # Extract standard reviews and paper contexts
        best_reviews = {}
        paper_contexts = {}
        for item in dataA:
            pred = ReviewProcessor.extract_review_content(item['pred_best_mode'])
            best_reviews[item['id']] = pred
            paper_contexts[item['id']] = item['paper_context']

        # Extract fast mode reviews
        standard_reviews = {}
        for item in dataB:
            standard_reviews[item['id']] = ReviewProcessor.extract_review_content(item[f'pred_fast_mode'])

        # Create comparison dataset
        comparison_data = []
        common_ids = set(best_reviews.keys()) & set(standard_reviews.keys())

        for paper_id in common_ids:
            comparison_data.append({
                'id': paper_id,
                'year': args.year,
                'paper_context': paper_contexts[paper_id],
                'DeepReviewer': best_reviews[paper_id],
                'other': standard_reviews[paper_id],
                'other_type': f'DeepReviewer_Standard'
            })

        return comparison_data


class EvaluationManager:
    """Manages the evaluation process using LLM API."""

    def __init__(self, args):
        """Initialize the evaluation manager."""
        # Set proxy settings
        os.environ['http_proxy'] = args.proxy
        os.environ['https_proxy'] = args.proxy

        # Initialize OpenAI client
        self.client = openai.Client(
            base_url=args.api_url,
            api_key=args.api_key
        )
        self.args = args

    def prepare_prompt(self, paper_item):
        """
        Prepare the prompt for the LLM evaluation.

        Args:
            paper_item: Dictionary containing paper and review data

        Returns:
            tuple: Prepared prompt and the ordering information
        """
        # Randomly determine the order of reviews (A or B)
        if random.randint(0, 1):
            content = (
                    '# Paper:\n' + paper_item['paper_context'] +
                    '\n\n---***---\n---***---\n---***---\n' +
                    '#Assistant A:\n' + paper_item['DeepReviewer'] +
                    '\n\n---***---\n---***---\n---***---\n' +
                    '#Assistant B:\n' + paper_item['other']
            )
            ordering = 'A'  # DeepReviewer is Assistant A
        else:
            content = (
                    '# Paper:\n' + paper_item['paper_context'] +
                    '\n\n---***---\n---***---\n---***---\n' +
                    '#Assistant A:\n' + paper_item['other'] +
                    '\n\n---***---\n---***---\n---***---\n' +
                    '#Assistant B:\n' + paper_item['DeepReviewer']
            )
            ordering = 'B'  # DeepReviewer is Assistant B

        return content, ordering

    def evaluate_reviews(self, paper_item):
        """
        Evaluate reviews using the LLM API.

        Args:
            paper_item: Dictionary containing paper and review data

        Returns:
            dict: Updated paper item with evaluation results
        """
        # Prepare the prompt and get ordering information
        content, ordering = self.prepare_prompt(paper_item)
        paper_item['v.s.'] = ordering

        # Call the API
        response = self.client.chat.completions.create(
            model=self.args.model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {'role': 'user', 'content': content},
            ],
            temperature=0,
            max_tokens=8192,
        )

        # Extract and store the result
        output = response.choices[0].message.content
        paper_item['result'] = output

        return paper_item


class ResultWriter:
    """Handles writing results to files."""

    @staticmethod
    def write_result(result, output_path):
        """
        Write a single result to the output file.

        Args:
            result: Result to write
            output_path: Path to write the result to
        """
        print(result)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

def print_result(data):
    def get_result(line):
        for l in line.split('\n'):
            if 'Better Assistant' in l:
                l = l.replace('Better', '').replace('Assistant', '')
                if 'Tie' in l:
                    return 'Tie'
                elif 'B' in l:
                    return 'B'
                else:
                    return 'A'
    results_Technical_Accuracy = []
    result_Constructive_Value = []
    result_Analytical_Depth = []
    result_Communication_Clarity = []
    result_Overall_Judgment = []
    for i in data:
        result = i['result']
        for line in result.split('\n\n'):
            if 'Better Assistant' in line:
                if 'Overall Judgment' in line:
                    r = get_result(line)
                    if r == i['v.s.']:
                        result_Overall_Judgment.append('win')
                    elif r == 'Tie':
                        result_Overall_Judgment.append('tie')
                    else:
                        result_Overall_Judgment.append('lose')

                elif 'Constructive Value' in line:
                    r = get_result(line)
                    if r == i['v.s.']:
                        result_Constructive_Value.append('win')
                    elif r == 'Tie':
                        result_Constructive_Value.append('tie')
                    else:
                        result_Constructive_Value.append('lose')
                elif 'Analytical Depth' in line:
                    r = get_result(line)
                    if r == i['v.s.']:
                        result_Analytical_Depth.append('win')
                    elif r == 'Tie':
                        result_Analytical_Depth.append('tie')
                    else:
                        result_Analytical_Depth.append('lose')
                elif 'Communication Clarity' in line:
                    r = get_result(line)
                    if r == i['v.s.']:
                        result_Communication_Clarity.append('win')
                    elif r == 'Tie':
                        result_Communication_Clarity.append('tie')
                    else:
                        result_Communication_Clarity.append('lose')
                elif 'Technical Accuracy' in line:
                    r = get_result(line)
                    if r == i['v.s.']:
                        results_Technical_Accuracy.append('win')
                    elif r == 'Tie':
                        results_Technical_Accuracy.append('tie')
                    else:
                        results_Technical_Accuracy.append('lose')

    print('Total Number', len(result_Overall_Judgment))
    print('Overall Judgment Win', result_Overall_Judgment.count('win') / len(result_Overall_Judgment))
    print('Overall Judgment Tie', result_Overall_Judgment.count('tie') / len(result_Overall_Judgment))
    print('Overall Judgment Lose', result_Overall_Judgment.count('lose') / len(result_Overall_Judgment))

    print('Constructive Value Win', result_Constructive_Value.count('win') / len(result_Constructive_Value))
    print('Constructive Value Tie', result_Constructive_Value.count('tie') / len(result_Constructive_Value))
    print('Constructive Value Lose', result_Constructive_Value.count('lose') / len(result_Constructive_Value))

    print('Analytical Depth Win', result_Analytical_Depth.count('win') / len(result_Analytical_Depth))
    print('Analytical Depth Tie', result_Analytical_Depth.count('tie') / len(result_Analytical_Depth))
    print('Analytical Depth Lose', result_Analytical_Depth.count('lose') / len(result_Analytical_Depth))

    print('Communication Clarity Win',
          result_Communication_Clarity.count('win') / len(result_Communication_Clarity))
    print('Communication Clarity Tie',
          result_Communication_Clarity.count('tie') / len(result_Communication_Clarity))
    print('Communication Clarity Lose',
          result_Communication_Clarity.count('lose') / len(result_Communication_Clarity))

    print('Technical Accuracy Win', results_Technical_Accuracy.count('win') / len(results_Technical_Accuracy))
    print('Technical Accuracy Tie', results_Technical_Accuracy.count('tie') / len(results_Technical_Accuracy))
    print('Technical Accuracy Lose', results_Technical_Accuracy.count('lose') / len(results_Technical_Accuracy))
    print()



def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Review Comparison System")

    # Proxy settings
    parser.add_argument("--proxy", type=str, default="", help="Proxy address")

    # API settings
    parser.add_argument("--api_key", type=str, default="your-api-key-here", help="API key for OpenAI")
    parser.add_argument("--api_url", type=str, default="https://api.openai.com/v1", help="OpenAI API base URL")

    # Model settings
    parser.add_argument("--model_name", type=str, default="gpt-4o-mini", help="Model name")

    # Paths
    parser.add_argument("--output_path", type=str, default="DeepReviewer_win_rate.json", help="Output file path")
    parser.add_argument("--sample_path", type=str, default="evaluate/DeepReview/sample.json", help="Sample file path")

    # Year
    parser.add_argument("--year", type=int, default=2024, help="Paper year")

    return parser.parse_args()

def main():
    """Main execution function."""
    args = parse_args()
    # Initialize components
    data_manager = DataManager()
    evaluator = EvaluationManager(args)

    # Load data
    dataA = data_manager.load_data(args.sample_path)
    dataB = data_manager.load_data(args.sample_path)

    # Prepare comparison data
    comparison_data = data_manager.prepare_comparison_data(dataA, dataB, args)

    # Evaluate all paper reviews
    results = []
    for paper_item in tqdm(comparison_data, desc="Evaluating reviews"): # TODO: use multi-thread to speed up
        record = {}
        # Get evaluation result
        result = evaluator.evaluate_reviews(paper_item)
        # record['v.s.'] = result['v.s.']
        # record['result'] = result['result']
        # results.append(record)
        # results.append(result)
        # Write result to file
        ResultWriter.write_result(result, args.output_path)
    print_result(results)

if __name__ == "__main__":
    main()

"""
# vllm 
python evaluate/DeepReview/rm_evaluate.py --api_url http://localhost:8000/v1 --model Qwen/Qwen2.5-7B-Instruct

python evaluate/DeepReview/rm_evaluate.py --api_url http://localhost:8000/v1 --model Qwen/Qwen3-4B

python evaluate/DeepReview/rm_evaluate.py --model_name gpt-4o
"""
