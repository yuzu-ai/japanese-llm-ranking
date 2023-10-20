""" Check agreement between two reviewers.
Usage: 
python check_reviewer_agreement.py data/rakuda_v2/model_judgment/claude-2_pair.jsonl data/rakuda_v2/model_judgment/gpt-4_pair.jsonl
"""
import argparse
import json

def compute_agreement(file1, file2):
    """
    Computes the fraction of times two files agree on the 'winner' of a matchup 
    with the same 'question_id', 'model1_id', and 'model2_id'.
    
    Args:
        file1 (str): Path to the first jsonl file.
        file2 (str): Path to the second jsonl file.
        
    Returns:
        float: Fraction of times the two files agree.
    """
    
    # Load the data from the files
    data1 = {}
    with open(file1, 'r') as f1:
        for line in f1:
            entry = json.loads(line)
            key = (entry["question_id"], entry["model1_id"], entry["model2_id"])
            data1[key] = entry["winner"]
            
    data2 = {}
    with open(file2, 'r') as f2:
        for line in f2:
            entry = json.loads(line)
            key = (entry["question_id"], entry["model1_id"], entry["model2_id"])
            data2[key] = entry["winner"]

    # Find overlapping keys
    overlapping_keys = set(data1.keys()).intersection(data2.keys())

    # Check agreement
    num_agreed = sum(1 for key in overlapping_keys if data1[key] == data2[key])
    
    return num_agreed / len(overlapping_keys) if overlapping_keys else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute agreement between two reviewers.")
    parser.add_argument('file1', type=str, help='Path to the first reviewer jsonl file.')
    parser.add_argument('file2', type=str, help='Path to the second reviewer jsonl file.')

    args = parser.parse_args()
    agreement = compute_agreement(args.file1, args.file2)
    print(f"Agreement fraction: {agreement:.3f}")