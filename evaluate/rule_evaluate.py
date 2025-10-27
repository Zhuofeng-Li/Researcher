import json
import argparse
import csv
import os
from datetime import datetime
from itertools import combinations

import torch
from scipy.stats import spearmanr
from sklearn.metrics import precision_recall_fscore_support


def get_pred(pred_context):
    """
    Extract and structure the prediction content from the model output.

    Args:
        pred_context: Raw prediction context from the model

    Returns:
        dict: Structured prediction with all review components
    """
    try:
        # Extract the review content from the boxed format
        pred_context = pred_context.split(r'\boxed_review{')[-1].split('\n}')[0]
    except:
        pred_context = pred_context['raw_text'].split(r'\boxed_review{')[-1].split('\n}')[0]

    # Initialize a dictionary with all review components
    pred = {
        'Summary': '',
        'Rating': '',
        'Soundness': '',
        'Presentation': '',
        'Contribution': '',
        'Strengths': '',
        'Weaknesses': '',
        'Suggestions': '',
        'Questions': '',
        'Confidence': '',
        'Decision': ''
    }

    # Parse each section from the prediction context
    for context in pred_context.split('## '):
        for key in pred:
            if key + ':\n\n' in context:
                pred[key] = context.split(key + ':\n\n')[-1].strip()

    return pred


def evaluate_deep_reviewer(data_path, mode='standard'):
    """
    Evaluate the performance of DeepReviewer model.

    Args:
        data_path: Path to the DeepReviewer predictions
        mode: Evaluation mode ('fast', 'standard', or 'best')

    Returns:
        dict: Results of the evaluation with various metrics
    """
    print(f"Evaluating DeepReviewer in {mode} mode...")

    # Load the DeepReviewer predictions
    with open(data_path, encoding='utf-8') as f:
        data = json.load(f)

    # Initialize lists for storing metrics
    # MSE and MAE metrics
    rating_mse_list_n = []
    rating_mae_list_n = []
    soundness_mse_list_n = []
    soundness_mae_list_n = []
    presentation_mse_list_n = []
    presentation_mae_list_n = []
    contribution_mse_list_n = []
    contribution_mae_list_n = []

    # For spearman correlation
    true_ratings = []
    pred_ratings = []
    true_soundness = []
    pred_soundness = []
    true_presentation = []
    pred_presentation = []
    true_contribution = []
    pred_contribution = []

    # For pairwise comparison
    paper_scores = []

    # For decision evaluation
    decision_acc = []
    true_decisions = []
    pred_decisions = []

    # Process each paper's review
    for item in data:
        # Parse the prediction in the specified mode
        pred = get_pred(item[f'pred_{mode}_mode'])
        item['pred'] = pred

        if 'pred' in item:
            try:
                # Skip if no rating is provided
                if item['pred']['Rating'] == '':
                    continue

                # Extract human review scores for this paper
                # Helper function to safely extract score
                def extract_score(value):
                    """Extract score from various formats (int, str, list)"""
                    if isinstance(value, (int, float)):
                        return int(value)
                    elif isinstance(value, str):
                        return int(value[0]) if value else 0
                    elif isinstance(value, (list, tuple)):
                        return int(value[0]) if value else 0
                    else:
                        return 0

                rates = torch.tensor([extract_score(r['content']['rating']) for r in item['review']], dtype=torch.float32)
                soundness = torch.tensor([extract_score(r['content']['soundness']) for r in item['review']],
                                         dtype=torch.float32)
                presentation = torch.tensor([extract_score(r['content']['presentation']) for r in item['review']],
                                            dtype=torch.float32)
                contribution = torch.tensor([extract_score(r['content']['contribution']) for r in item['review']],
                                            dtype=torch.float32)

                # Extract model predictions (handling possible newlines in output)
                estimated_score = torch.tensor(float(item['pred']['Rating'].split('\n')[0]))
                estimated_soundness = torch.tensor(float(item['pred']['Soundness'].split('\n')[0]))
                estimated_presentation = torch.tensor(float(item['pred']['Presentation'].split('\n')[0]))
                estimated_contribution = torch.tensor(float(item['pred']['Contribution'].split('\n')[0]))

                # Calculate average human scores as ground truth proxies
                true_rate_proxy_n = rates.mean()
                true_soundness_proxy_n = soundness.mean()
                true_presentation_proxy_n = presentation.mean()
                true_contribution_proxy_n = contribution.mean()

                # Store scores for pairwise comparison
                paper_scores.append({
                    'true_rating': true_rate_proxy_n.item(),
                    'pred_rating': estimated_score.item(),
                    'true_soundness': true_soundness_proxy_n.item(),
                    'pred_soundness': estimated_soundness.item(),
                    'true_presentation': true_presentation_proxy_n.item(),
                    'pred_presentation': estimated_presentation.item(),
                    'true_contribution': true_contribution_proxy_n.item(),
                    'pred_contribution': estimated_contribution.item()
                })

                # Store values for correlations and other metrics
                true_ratings.append(true_rate_proxy_n.item())
                pred_ratings.append(estimated_score.item())
                true_soundness.append(true_soundness_proxy_n.item())
                pred_soundness.append(estimated_soundness.item())
                true_presentation.append(true_presentation_proxy_n.item())
                pred_presentation.append(estimated_presentation.item())
                true_contribution.append(true_contribution_proxy_n.item())
                pred_contribution.append(estimated_contribution.item())

                # Calculate MSE and MAE for each metric
                rating_mse_list_n.append(torch.pow(estimated_score - true_rate_proxy_n, 2).item())
                rating_mae_list_n.append(torch.abs(estimated_score - true_rate_proxy_n).item())
                soundness_mse_list_n.append(torch.pow(estimated_soundness - true_soundness_proxy_n, 2).item())
                soundness_mae_list_n.append(torch.abs(estimated_soundness - true_soundness_proxy_n).item())
                presentation_mse_list_n.append(torch.pow(estimated_presentation - true_presentation_proxy_n, 2).item())
                presentation_mae_list_n.append(torch.abs(estimated_presentation - true_presentation_proxy_n).item())
                contribution_mse_list_n.append(torch.pow(estimated_contribution - true_contribution_proxy_n, 2).item())
                contribution_mae_list_n.append(torch.abs(estimated_contribution - true_contribution_proxy_n).item())

                # Process decision metrics
                true_decision = item['decision'].lower()
                pred_decision = item['pred']['Decision'].lower().strip()

                # Normalize decision to accept/reject
                if 'accept' in pred_decision.lower():
                    pred_decision = 'accept'
                else:
                    pred_decision = 'reject'

                # Convert to binary for F1 calculation
                if 'accept' in pred_decision:
                    pred_decisions.append(1)
                else:
                    pred_decisions.append(0)
                if 'accept' in true_decision:
                    true_decisions.append(1)
                else:
                    true_decisions.append(0)

                # Calculate decision accuracy
                if pred_decision in true_decision:
                    decision_acc.append(1.)
                else:
                    decision_acc.append(0.)

            except Exception as e:
                print(f"Error processing item: {e}")
                pass

    # Calculate pairwise comparison accuracy
    pairwise_accuracies = calculate_pairwise_accuracies(paper_scores)

    # Calculate mean values for all metrics
    rating_mse_n = torch.mean(torch.tensor(rating_mse_list_n)).item()
    rating_mae_n = torch.mean(torch.tensor(rating_mae_list_n)).item()
    soundness_mse_n = torch.mean(torch.tensor(soundness_mse_list_n)).item()
    soundness_mae_n = torch.mean(torch.tensor(soundness_mae_list_n)).item()
    presentation_mse_n = torch.mean(torch.tensor(presentation_mse_list_n)).item()
    presentation_mae_n = torch.mean(torch.tensor(presentation_mae_list_n)).item()
    contribution_mse_n = torch.mean(torch.tensor(contribution_mse_list_n)).item()
    contribution_mae_n = torch.mean(torch.tensor(contribution_mae_list_n)).item()

    # Calculate Spearman correlations
    rating_spearman_val, _ = spearmanr(true_ratings, pred_ratings)
    soundness_spearman_val, _ = spearmanr(true_soundness, pred_soundness)
    presentation_spearman_val, _ = spearmanr(true_presentation, pred_presentation)
    contribution_spearman_val, _ = spearmanr(true_contribution, pred_contribution)

    # Format Spearman values
    rating_spearman = f"{rating_spearman_val:.4f}" if not isinstance(rating_spearman_val, float) or not (
                rating_spearman_val is None) else 'nan'
    soundness_spearman = f"{soundness_spearman_val:.4f}" if not isinstance(soundness_spearman_val, float) or not (
                soundness_spearman_val is None) else 'nan'
    presentation_spearman = f"{presentation_spearman_val:.4f}" if not isinstance(presentation_spearman_val,
                                                                                 float) or not (
                presentation_spearman_val is None) else 'nan'
    contribution_spearman = f"{contribution_spearman_val:.4f}" if not isinstance(contribution_spearman_val,
                                                                                 float) or not (
                contribution_spearman_val is None) else 'nan'

    # Calculate decision metrics
    decision_acc_val = torch.mean(torch.tensor(decision_acc)).item()
    _, _, f1_val, _ = precision_recall_fscore_support(true_decisions, pred_decisions, average='macro')
    f1_score = float(f1_val)

    # Prepare results dictionary
    results = {
        'Model/Method': f'DeepReviewer {mode} (n={len(data)})',
        'Rating MSE': f"{rating_mse_n:.4f}",
        'Rating MAE': f"{rating_mae_n:.4f}",
        'Soundness MSE': f"{soundness_mse_n:.4f}",
        'Soundness MAE': f"{soundness_mae_n:.4f}",
        'Presentation MSE': f"{presentation_mse_n:.4f}",
        'Presentation MAE': f"{presentation_mae_n:.4f}",
        'Contribution MSE': f"{contribution_mse_n:.4f}",
        'Contribution MAE': f"{contribution_mae_n:.4f}",
        'Rating Spearman': rating_spearman,
        'Soundness Spearman': soundness_spearman,
        'Presentation Spearman': presentation_spearman,
        'Contribution Spearman': contribution_spearman,
        'Decision Accuracy': f"{decision_acc_val:.4f}",
        'Decision F1': f"{f1_score:.4f}",
        'Pairwise Rating Acc': f"{pairwise_accuracies['rating']:.4f}",
        'Pairwise Soundness Acc': f"{pairwise_accuracies['soundness']:.4f}",
        'Pairwise Presentation Acc': f"{pairwise_accuracies['presentation']:.4f}",
        'Pairwise Contribution Acc': f"{pairwise_accuracies['contribution']:.4f}",
    }

    return results


def calculate_pairwise_accuracies(paper_scores):
    """
    Calculate pairwise accuracy for each metric by comparing rankings.

    Args:
        paper_scores: List of dictionaries containing true and predicted scores

    Returns:
        dict: Pairwise accuracies for each metric
    """
    total_pairs = 0
    correct_pairs = {
        'rating': 0,
        'soundness': 0,
        'presentation': 0,
        'contribution': 0
    }

    # Get all possible paper pairs using combinations
    for paper1, paper2 in combinations(paper_scores, 2):
        total_pairs += 1

        # Check if model correctly identifies which paper has higher rating
        true_rating_order = paper1['true_rating'] > paper2['true_rating']
        pred_rating_order = paper1['pred_rating'] > paper2['pred_rating']
        if true_rating_order == pred_rating_order:
            correct_pairs['rating'] += 1

        # Check soundness ranking
        true_soundness_order = paper1['true_soundness'] > paper2['true_soundness']
        pred_soundness_order = paper1['pred_soundness'] > paper2['pred_soundness']
        if true_soundness_order == pred_soundness_order:
            correct_pairs['soundness'] += 1

        # Check presentation ranking
        true_presentation_order = paper1['true_presentation'] > paper2['true_presentation']
        pred_presentation_order = paper1['pred_presentation'] > paper2['pred_presentation']
        if true_presentation_order == pred_presentation_order:
            correct_pairs['presentation'] += 1

        # Check contribution ranking
        true_contribution_order = paper1['true_contribution'] > paper2['true_contribution']
        pred_contribution_order = paper1['pred_contribution'] > paper2['pred_contribution']
        if true_contribution_order == pred_contribution_order:
            correct_pairs['contribution'] += 1

    # Calculate accuracies as percentage of correct pairs
    pairwise_accuracies = {
        metric: count / total_pairs if total_pairs > 0 else 0
        for metric, count in correct_pairs.items()
    }

    return pairwise_accuracies


def create_markdown_table(results):
    """
    Create a markdown table for displaying evaluation results.

    Args:
        results: Dictionary containing evaluation results

    Returns:
        str: Markdown table representation
    """
    # Define table header
    header = "| Metric | Value |\n|---|---|\n"

    # Add all results to table
    table_rows = []
    for key, value in results.items():
        if key != "Model/Method":  # Skip the model name
            table_rows.append(f"| {key} | {value} |")

    # Join all rows and return complete table
    return header + "\n".join(table_rows)


def main():
    """
    Main function to run the evaluation.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluate DeepReviewer predictions')
    parser.add_argument('data_path', type=str, help='Path to the JSON file containing predictions')
    parser.add_argument('--modes', nargs='+', default=['fast'],
                        choices=['fast', 'standard', 'best'],
                        help='Evaluation modes to run (default: all modes)')

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    output_dir = 'evaluate/result'
    os.makedirs(output_dir, exist_ok=True)

    # Collect all results for CSV export
    all_results = []

    # Evaluate specified modes and display results
    for mode in args.modes:
        results = evaluate_deep_reviewer(args.data_path, mode)
        print(f"\nResults for DeepReviewer in {mode} mode:")
        markdown_table = create_markdown_table(results)

        print(markdown_table)

        # Store results for CSV
        all_results.append(results)

    # Save results to CSV with selected metrics only
    # Generate filename with current timestamp
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    csv_filename = os.path.join(output_dir, f'rule_results_{timestamp}.csv')

    # Define which metrics to include in CSV
    selected_metrics = [
        'Model/Method',
        'Rating MSE',
        'Soundness MSE',
        'Presentation MSE',
        'Contribution MSE',
        'Decision F1',
        'Pairwise Rating Acc'
    ]

    with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
        if all_results:
            writer = csv.DictWriter(csvfile, fieldnames=selected_metrics)
            writer.writeheader()

            for result in all_results:
                # Filter to only include selected metrics
                filtered_result = {k: result[k] for k in selected_metrics if k in result}
                writer.writerow(filtered_result)

    print(f"Results saved to {csv_filename}")


if __name__ == "__main__":
    main()

"""
python evaluate/rule_evaluate.py evaluate/review/benchmark_deepreviewer_results.json --modes fast
"""

# 