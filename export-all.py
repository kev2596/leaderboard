import subprocess
import json
import re
import os
import shutil
import csv
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np

# ==================== KONFIGURATION ====================
class Config:
    RCLONE = Path(r"C:\Users\kevin.haizmann\Downloads\rclone-v1.71.2-windows-amd64\rclone-v1.71.2-windows-amd64\rclone.exe")
    GIT = Path(r"C:\Program Files\Git\bin\git.exe")
    REMOTE = "switch:"
    LOCAL_ROOT = Path(r"C:\Users\kevin.haizmann\OneDrive - OST\Dokumente\Switch\Exports")
    SOLUTION_DIR = LOCAL_ROOT / "solution"
    GITHUB_REPO = Path(r"C:\Users\kevin.haizmann\OneDrive - OST\Dokumente\leaderboard")
    ENABLE_GIT_PUSH = True
    UPDATE_INTERVAL_HOURS = 1
    PARTICIPANT_PATTERN = re.compile(r"PARTICIPANT_\d{1,3}")
    SUBMISSION_PATTERN = re.compile(r"Results_(\d+)_(\d+)\.csv", re.IGNORECASE)

# ==================== LOGGING ====================
def log(message: str, level: str = "INFO"):
    """Unified logging with emoji prefixes."""
    emoji_map = {
        "INFO": "ℹ️", "SUCCESS": "✅", "WARNING": "⚠️", 
        "ERROR": "❌", "PROCESS": "🔄", "DATA": "📊"
    }
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {emoji_map.get(level, '•')} {message}")

# ==================== RCLONE OPERATIONS ====================
def run_command(cmd: List[str], check: bool = False) -> Tuple[int, str, str]:
    """Execute shell command and return output."""
    result = subprocess.run(cmd, capture_output=True, text=True)
    if check and result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, cmd, result.stderr)
    return result.returncode, result.stdout, result.stderr

def get_remote_directories() -> List[str]:
    """Fetch all directories from remote."""
    try:
        rc, out, err = run_command([str(Config.RCLONE), "lsjson", Config.REMOTE, "-R", "--dirs-only"])
        if rc != 0:
            log(f"Failed to fetch directories: {err}", "ERROR")
            return []
        return [item["Path"].replace("\\", "/") for item in json.loads(out) if "Path" in item]
    except Exception as e:
        log(f"Error parsing rclone output: {e}", "ERROR")
        return []

def find_participant_directories(all_dirs: List[str]) -> List[str]:
    """Extract participant base directories."""
    participants = set()
    for path in all_dirs:
        parts = path.split("/")
        for i, segment in enumerate(parts):
            if Config.PARTICIPANT_PATTERN.fullmatch(segment):
                participants.add("/".join(parts[:i + 1]))
                break
    return sorted(participants)

def sync_submissions(participant_path: str) -> bool:
    """Download submissions for a participant."""
    remote_path = f"{Config.REMOTE}{participant_path}/Submissions"
    local_path = Config.LOCAL_ROOT / participant_path.replace("/", os.sep) / "Submissions"
    
    log(f"Syncing: {participant_path}")
    cmd = [str(Config.RCLONE), "copy", remote_path, str(local_path), "--update", "--create-empty-src-dirs"]
    rc, _, err = run_command(cmd)
    
    if rc == 0:
        return True
    elif "not found" in err.lower():
        log(f"No submissions folder for {participant_path}", "WARNING")
    else:
        log(f"Sync failed for {participant_path}: {err}", "ERROR")
    return False

# ==================== DATA PROCESSING ====================
def load_csv_data(filepath: Path) -> Optional[np.ndarray]:
    """Load numeric data from CSV, handling various formats."""
    try:
        for kwargs in [
            {"delimiter": ",", "skiprows": 1},
            {"skiprows": 1},
            {"delimiter": ","},
            {}
        ]:
            try:
                arr = np.loadtxt(filepath, **kwargs)
                if arr.ndim > 1 and arr.shape[1] >= 2:
                    arr = arr[:, 1]  # Extract value column
                elif arr.ndim > 1:
                    arr = arr.flatten()
                return arr.astype(float)
            except:
                continue
        return None
    except Exception as e:
        log(f"Cannot read {filepath.name}: {e}", "WARNING")
        return None

def load_solutions() -> Dict[str, np.ndarray]:
    """Load all solution files."""
    solutions = {}
    if not Config.SOLUTION_DIR.exists():
        log(f"Solution directory missing: {Config.SOLUTION_DIR}", "ERROR")
        return solutions
    
    for file in Config.SOLUTION_DIR.glob("*.csv"):
        data = load_csv_data(file)
        if data is not None:
            solutions[file.name] = data
    
    log(f"Loaded {len(solutions)} solution file(s)", "DATA")
    return solutions

def compute_rmse(prediction: np.ndarray, truth: np.ndarray) -> Optional[float]:
    """Calculate RMSE between two arrays."""
    if prediction is None or truth is None or len(prediction) == 0 or len(truth) == 0:
        return None
    
    length = min(len(prediction), len(truth))
    mse = np.mean((prediction[:length] - truth[:length]) ** 2)
    return float(np.sqrt(mse))

# ==================== EVALUATION ====================
def evaluate_all_submissions(solutions: Dict[str, np.ndarray]) -> List[Dict]:
    """Evaluate all participant submissions."""
    results = []
    
    for participant_dir in Config.LOCAL_ROOT.rglob("PARTICIPANT_*"):
        if not participant_dir.is_dir():
            continue
        
        submissions_dir = participant_dir / "Submissions"
        if not submissions_dir.exists():
            continue
        
        for submission_file in submissions_dir.glob("Results_*.csv"):
            match = Config.SUBMISSION_PATTERN.match(submission_file.name)
            if not match:
                continue
            
            participant_id, submission_num = match.groups()
            solution_name = next(iter(solutions.keys()), None)  # Use first solution
            
            if solution_name not in solutions:
                continue
            
            submission_data = load_csv_data(submission_file)
            if submission_data is None:
                continue
            
            rmse = compute_rmse(submission_data, solutions[solution_name])
            if rmse is None:
                continue
            
            results.append({
                "rank": 0,  # Will be assigned after sorting
                "submission_id": f"PARTICIPANT_{participant_id}_Sub{submission_num}",
                "participant_id": participant_id,
                "submission_num": int(submission_num),
                "rmse": rmse,
                "filename": submission_file.name,
                "path": str(participant_dir)
            })
    
    # Sort by RMSE and assign ranks
    results.sort(key=lambda x: x["rmse"])
    for i, result in enumerate(results, start=1):
        result["rank"] = i
    
    log(f"Evaluated {len(results)} submission(s)", "DATA")
    return results

def save_ranking_csv(results: List[Dict], output_path: Path):
    """Save ranking results to CSV."""
    try:
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
            writer.writerow(["Rank", "Submission_ID", "Participant_ID", "Submission_Num", "RMSE", "Filename", "Pfad"])
            for result in results:
                writer.writerow([
                    result["rank"],
                    result["submission_id"],
                    result["participant_id"],
                    result["submission_num"],
                    f"{result['rmse']:.6f}",
                    result["filename"],
                    result["path"]
                ])
        log(f"Ranking saved to {output_path.name}", "SUCCESS")
    except Exception as e:
        log(f"Failed to save ranking: {e}", "ERROR")

# ==================== GIT OPERATIONS ====================
def push_to_github():
    """Push updated CSV to GitHub."""
    if not Config.ENABLE_GIT_PUSH:
        log("Git push disabled", "INFO")
        return False
    
    if not Config.GITHUB_REPO.exists():
        log(f"GitHub repo path missing: {Config.GITHUB_REPO}", "ERROR")
        return False
    
    try:
        # Copy CSV to repo
        csv_source = Config.LOCAL_ROOT / "rmse_ranking.csv"
        csv_dest = Config.GITHUB_REPO / "rmse_ranking.csv"
        shutil.copy2(csv_source, csv_dest)
        log("CSV copied to GitHub repo", "PROCESS")
        
        # Git operations
        os.chdir(Config.GITHUB_REPO)
        run_command([str(Config.GIT), "add", "rmse_ranking.csv"], check=True)
        
        commit_msg = f"Update rankings - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        rc, out, _ = run_command([str(Config.GIT), "commit", "-m", commit_msg])
        
        if "nothing to commit" in out:
            log("No changes to commit", "INFO")
            return True
        
        run_command([str(Config.GIT), "push"], check=True)
        log("Pushed to GitHub successfully", "SUCCESS")
        return True
        
    except subprocess.CalledProcessError as e:
        log(f"Git error: {e}", "ERROR")
        return False
    except Exception as e:
        log(f"Push failed: {e}", "ERROR")
        return False

# ==================== MAIN WORKFLOW ====================
def run_update_cycle() -> bool:
    """Execute complete update cycle."""
    log("="*60)
    log("Starting update cycle", "PROCESS")
    log("="*60)
    
    # 1. Fetch remote directories
    all_dirs = get_remote_directories()
    if not all_dirs:
        log("No directories found on remote", "WARNING")
        return False
    
    # 2. Find and sync participant submissions
    participants = find_participant_directories(all_dirs)
    log(f"Found {len(participants)} participant(s)")
    
    dirs_set = set(all_dirs)
    for participant in participants:
        if f"{participant}/Submissions" in dirs_set:
            sync_submissions(participant)
    
    # 3. Load solutions
    solutions = load_solutions()
    if not solutions:
        log("No solution files found", "ERROR")
        return False
    
    # 4. Evaluate submissions
    results = evaluate_all_submissions(solutions)
    if not results:
        log("No valid submissions found", "WARNING")
        return False
    
    # 5. Save ranking
    output_path = Config.LOCAL_ROOT / "rmse_ranking.csv"
    save_ranking_csv(results, output_path)
    
    # 6. Push to GitHub
    push_to_github()
    
    log("Update cycle completed", "SUCCESS")
    log("="*60)
    return True

def main():
    """Main loop for continuous updates."""
    log("🤖 Automatic Leaderboard Update Started")
    log(f"⏰ Update Interval: Every {Config.UPDATE_INTERVAL_HOURS} hour(s)")
    log(f"📁 Local Root: {Config.LOCAL_ROOT}")
    log(f"🌐 GitHub Repo: {Config.GITHUB_REPO}")
    log(f"🔄 Git Push: {'Enabled' if Config.ENABLE_GIT_PUSH else 'Disabled'}")
    
    while True:
        try:
            run_update_cycle()
            
            sleep_seconds = Config.UPDATE_INTERVAL_HOURS * 3600
            log(f"Sleeping for {Config.UPDATE_INTERVAL_HOURS} hour(s)...")
            time.sleep(sleep_seconds)
            
        except KeyboardInterrupt:
            log("Program stopped by user", "WARNING")
            break
        except Exception as e:
            log(f"Unexpected error: {e}", "ERROR")
            log("Retrying in 5 minutes...")
            time.sleep(300)

if __name__ == "__main__":
    main()