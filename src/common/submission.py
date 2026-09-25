import os
import subprocess
import pandas as pd
from src.common.paths import REPO_ROOT, TEST_DIR

def write_submission_files(scored_pairs_df: pd.DataFrame, threshold: float, required_s1_ids: set, candidate_pairs_df: pd.DataFrame, output_dir: str = "output"):
    """
    Writes matching_results.tsv and candidate_pairs.tsv from scored pairs and candidate pairs.
    """
    os.makedirs(output_dir, exist_ok=True)
    matching_path = os.path.join(output_dir, "matching_results.tsv")
    candidate_path = os.path.join(output_dir, "candidate_pairs.tsv")

    # 1. Matching Results
    if not scored_pairs_df.empty:
        matches_df = scored_pairs_df[scored_pairs_df['score'] >= threshold].copy()
        if not matches_df.empty:
            matching_agg = matches_df.groupby('s1_id')['cand_id'].apply(lambda x: ','.join(x.dropna().unique())).reset_index()
            matching_agg.rename(columns={'s1_id': 'source1_entity_id', 'cand_id': 'matched_entity_ids'}, inplace=True)
        else:
            matching_agg = pd.DataFrame(columns=['source1_entity_id', 'matched_entity_ids'])
    else:
        matching_agg = pd.DataFrame(columns=['source1_entity_id', 'matched_entity_ids'])
    
    required_df = pd.DataFrame({'source1_entity_id': list(required_s1_ids)})
    final_matching = pd.merge(required_df, matching_agg, on='source1_entity_id', how='left')
    final_matching['matched_entity_ids'] = final_matching['matched_entity_ids'].fillna('')
    
    final_matching.to_csv(matching_path, sep='\t', index=False)

    # 2. Candidate Pairs
    if not candidate_pairs_df.empty:
        candidate_agg = candidate_pairs_df.groupby('s1_id')['cand_id'].apply(lambda x: ','.join(x.dropna().unique())).reset_index()
        candidate_agg.rename(columns={'s1_id': 'source1_entity_id', 'cand_id': 'candidate_entity_ids'}, inplace=True)
    else:
        candidate_agg = pd.DataFrame(columns=['source1_entity_id', 'candidate_entity_ids'])

    final_candidate = pd.merge(required_df, candidate_agg, on='source1_entity_id', how='left')
    final_candidate['candidate_entity_ids'] = final_candidate['candidate_entity_ids'].fillna('')
    
    final_candidate.to_csv(candidate_path, sep='\t', index=False)

    return matching_path, candidate_path

def validate_submission(matching_path: str, candidate_path: str, test_dir: str = None, check_ids: bool = False):
    """
    Runs the official validator script.
    """
    if test_dir is None:
        test_dir = str(TEST_DIR)

    script_path = str(REPO_ROOT / "utils" / "validate_submission.py")
    cmd = [
        "python", script_path,
        "--matching", matching_path,
        "--candidate", candidate_path,
        "--test-dir", test_dir
    ]
    if check_ids:
        cmd.append("--check-ids")
        
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
        
    return result.returncode == 0
