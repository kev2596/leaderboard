import subprocess
import json
import re
import os
import numpy as np
import csv
import time
from datetime import datetime

# ---- Einstellungen ----
RCLONE = r"C:\Users\kevin.haizmann\Downloads\rclone-v1.71.2-windows-amd64\rclone-v1.71.2-windows-amd64\rclone.exe"
REMOTE = "switch:"
LOCAL_ROOT = r"C:\Users\kevin.haizmann\OneDrive - OST\Dokumente\Switch\Exports"
SOLUTION_DIR = os.path.join(LOCAL_ROOT, "solution")
GIT = r"C:\Program Files\Git\bin\git.exe"

# GitHub Settings
GITHUB_REPO_PATH = r"C:\Users\kevin.haizmann\OneDrive - OST\Dokumente\leaderboard"
ENABLE_GIT_PUSH = True

# ---- Regex für Participant-Ordner ----
PART_REGEX = re.compile(r"PARTICIPANT_\d{1,3}")

# ---- Logging ----
def log(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

# ---- Hilfsfunktion zum Ausführen von rclone ----
def run_rclone(cmd):
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode, result.stdout, result.stderr

# ---- Alle Verzeichnisse auf dem Remote abrufen ----
def get_all_remote_dirs():
    cmd = [RCLONE, "lsjson", REMOTE, "-R", "--dirs-only"]
    rc, out, err = run_rclone(cmd)
    if rc != 0:
        log(f"❌ Fehler beim Abrufen der Verzeichnisse: {err}")
        return []
    try:
        items = json.loads(out)
        return [item["Path"].replace("\\", "/") for item in items if "Path" in item]
    except Exception as e:
        log(f"❌ Fehler beim Parsen von rclone-Ausgabe: {e}")
        return []

# ---- Participant-Ordner finden ----
def find_participant_bases(dir_paths):
    participant_paths = set()
    for p in dir_paths:
        parts = p.split("/")
        for i, seg in enumerate(parts):
            if PART_REGEX.fullmatch(seg):
                base = "/".join(parts[:i + 1])
                participant_paths.add(base)
                break
    return sorted(participant_paths)

# ---- Submissions kopieren ----
def copy_submissions(participant_path):
    remote_sub = f"{REMOTE}{participant_path}/Submissions"
    local_sub = os.path.join(LOCAL_ROOT, *participant_path.split("/"), "Submissions")

    log(f"📥 Kopiere: {remote_sub}")
    cmd = [
        RCLONE, "copy", remote_sub, local_sub,
        "--update", "--create-empty-src-dirs"
    ]
    rc, out, err = run_rclone(cmd)

    if rc == 0:
        log(f"✅ {participant_path}")
    else:
        if "not found" in (err + out).lower():
            log(f"⚠️ Kein Submissions-Ordner für {participant_path}")
        else:
            log(f"❌ Fehler bei {participant_path}: {err.strip()}")

# ---- Dateien einlesen ----
def load_numeric_file(path):
    try:
        try:
            arr = np.loadtxt(path, delimiter=",", skiprows=1)
        except Exception:
            try:
                arr = np.loadtxt(path, skiprows=1)
            except Exception:
                try:
                    arr = np.loadtxt(path, delimiter=",")
                except Exception:
                    arr = np.loadtxt(path)
        
        if arr.ndim > 1:
            if arr.shape[1] >= 2:
                arr = arr[:, 1]
            else:
                arr = arr.flatten()
        
        return np.asarray(arr, dtype=float)
    except Exception as e:
        log(f"⚠️ Datei nicht numerisch lesbar: {path} — {e}")
        return None

def parse_submission_filename(fname):
    pattern = r"Results_(\d+)_(\d+)\.csv"
    match = re.match(pattern, fname, re.IGNORECASE)
    if match:
        return match.group(1), int(match.group(2))
    return None, None

def match_solution_file(submission_fname, solutions):
    if solutions:
        return list(solutions.keys())[0]
    return None

def load_solutions(solution_dir):
    solutions = {}
    if not os.path.isdir(solution_dir):
        log(f"❌ Solution-Ordner existiert nicht: {solution_dir}")
        return solutions
    for fname in os.listdir(solution_dir):
        fpath = os.path.join(solution_dir, fname)
        if os.path.isfile(fpath):
            arr = load_numeric_file(fpath)
            if arr is not None:
                solutions[fname] = arr
    log(f"🔢 Geladene Lösungsdateien: {len(solutions)}")
    return solutions

def find_local_participants(local_root):
    matches = []
    for root, dirs, _ in os.walk(local_root):
        for d in dirs:
            if PART_REGEX.fullmatch(d):
                matches.append(os.path.join(root, d))
    matches = sorted(set(matches))
    log(f"👥 Gefundene lokale Participant-Ordner: {len(matches)}")
    return matches

def compute_rmse(a, b):
    if a is None or b is None:
        return None
    la, lb = len(a), len(b)
    if la == 0 or lb == 0:
        return None
    if la != lb:
        L = min(la, lb)
        a2, b2 = a[:L], b[:L]
    else:
        a2, b2 = a, b
    mse = np.mean((a2 - b2) ** 2)
    return float(np.sqrt(mse))

# ---- Evaluation aller Teilnehmer ----
def evaluate_participants(solutions, local_root, output_summary_csv=None, output_rank_csv=None):
    participants = find_local_participants(local_root)
    if not participants:
        log("⚠️ Keine Teilnehmerordner gefunden.")
        return

    summary = []
    submission_stats = []

    for ppath in participants:
        submissions_dir = os.path.join(ppath, "Submissions")
        if not os.path.isdir(submissions_dir):
            continue

        file_names = [f for f in os.listdir(submissions_dir)
                      if os.path.isfile(os.path.join(submissions_dir, f))]
        
        for fname in file_names:
            participant_id, submission_num = parse_submission_filename(fname)
            if participant_id is None:
                continue
            
            sol_name = match_solution_file(fname, solutions)
            if sol_name is None or sol_name not in solutions:
                continue
            
            sol_arr = solutions[sol_name]
            part_arr = load_numeric_file(os.path.join(submissions_dir, fname))
            
            if part_arr is None:
                continue
            
            rmse = compute_rmse(sol_arr, part_arr)
            if rmse is None:
                continue
            
            submission_id = f"PARTICIPANT_{participant_id}_Sub{submission_num}"
            
            summary.append({
                "participant": os.path.basename(ppath),
                "participant_id": participant_id,
                "submission_num": submission_num,
                "submission_id": submission_id,
                "participant_path": ppath,
                "file": fname,
                "solution_file": sol_name,
                "rmse": rmse
            })
            
            submission_stats.append({
                "submission_id": submission_id,
                "participant": os.path.basename(ppath),
                "participant_id": participant_id,
                "submission_num": submission_num,
                "rmse": rmse,
                "file": fname,
                "participant_path": ppath
            })

    if not submission_stats:
        log("⚠️ Keine gültigen Submissions gefunden.")
        return {}, summary

    ranked = sorted(submission_stats, key=lambda x: x["rmse"])
    log(f"🏆 Rangliste erstellt: {len(ranked)} Submissions")

    # CSV: Detail-Summary
    if output_summary_csv:
        try:
            with open(output_summary_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=[
                    "participant", "participant_id", "submission_num", "submission_id",
                    "participant_path", "file", "solution_file", "rmse"
                ])
                writer.writeheader()
                for row in summary:
                    writer.writerow(row)
            log(f"📄 Detail-Summary gespeichert")
        except Exception as e:
            log(f"❌ Fehler beim Schreiben der Detail-CSV: {e}")

    # CSV: Ranking
    if output_rank_csv:
        try:
            with open(output_rank_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
                writer.writerow(["Rank", "Submission_ID", "Participant_ID", "Submission_Num", 
                               "RMSE", "Filename", "Pfad"])
                for i, sub in enumerate(ranked, start=1):
                    writer.writerow([
                        i,
                        sub["submission_id"],
                        sub["participant_id"],
                        sub["submission_num"],
                        f"{sub['rmse']:.6f}",
                        sub["file"],
                        sub["participant_path"]
                    ])
            log(f"🏁 Rangliste gespeichert")
        except Exception as e:
            log(f"❌ Fehler beim Schreiben der Ranglisten-CSV: {e}")

    return submission_stats, summary

# ---- Git Push zum GitHub ----
def git_push_updates():
    if not ENABLE_GIT_PUSH:
        log("ℹ️ Git-Push ist deaktiviert")
        return False
    
    if not os.path.exists(GITHUB_REPO_PATH):
        log(f"❌ GitHub Repo-Pfad existiert nicht: {GITHUB_REPO_PATH}")
        return False
    
    try:
        log("📤 Kopiere CSV zu GitHub Repo...")
        
        import shutil
        csv_source = os.path.join(LOCAL_ROOT, "rmse_ranking.csv")
        csv_dest = os.path.join(GITHUB_REPO_PATH, "rmse_ranking.csv")
        shutil.copy2(csv_source, csv_dest)
        
        os.chdir(GITHUB_REPO_PATH)
        
        subprocess.run([GIT, "add", "rmse_ranking.csv"], check=True)

        commit_msg = f"Update rankings - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        result = subprocess.run([GIT, "commit", "-m", commit_msg], capture_output=True, text=True)

        # Check if there were changes
        if "nothing to commit" in result.stdout:
            log("ℹ️ Keine Änderungen zu committen")
            return True
        
        # Push
        subprocess.run([GIT, "push"], check=True)
        log("✅ Erfolgreich zu GitHub gepusht!")
        return True
        
    except subprocess.CalledProcessError as e:
        log(f"❌ Git-Fehler: {e}")
        return False
    except Exception as e:
        log(f"❌ Fehler beim Git-Push: {e}")
        return False

# ---- Hauptprogramm ----
def run_update_cycle():
    log("=" * 60)
    log("🚀 Starte Update-Zyklus")
    log("=" * 60)
    
    if not os.path.exists(LOCAL_ROOT):
        log(f"❌ LOCAL_ROOT existiert nicht: {LOCAL_ROOT}")
        return False

    # 1. Remote-Daten herunterladen
    log("🔍 Suche nach PARTICIPANT-Ordnern auf Remote...")
    dirs = get_all_remote_dirs()
    if not dirs:
        log("⚠️ Keine Verzeichnisse gefunden")
        return False

    participant_bases = find_participant_bases(dirs)
    log(f"📁 Gefundene Participant-Ordner: {len(participant_bases)}")

    dirs_set = set(dirs)
    for base in participant_bases:
        if f"{base}/Submissions" in dirs_set:
            copy_submissions(base)

    log("✅ Download abgeschlossen")

    # 2. Lösungen laden
    solutions = load_solutions(SOLUTION_DIR)
    if not solutions:
        log("❌ Keine Lösungen gefunden")
        return False

    # 3. Evaluation durchführen
    log("📊 Starte Evaluation...")
    out_summary = os.path.join(LOCAL_ROOT, "rmse_summary.csv")
    out_rank = os.path.join(LOCAL_ROOT, "rmse_ranking.csv")
    evaluate_participants(solutions, LOCAL_ROOT,
                          output_summary_csv=out_summary,
                          output_rank_csv=out_rank)

    # 4. Zu GitHub pushen
    if ENABLE_GIT_PUSH:
        log("🔄 Push zu GitHub...")
        git_push_updates()

    log("✅ Update-Zyklus abgeschlossen")
    log("=" * 60)
    return True

# ---- Hauptschleife für stündliche Updates ----
def main():
    log("🤖 Automatisches Leaderboard-Update gestartet")
    log(f"⏰ Update-Intervall: jede Stunde")
    log(f"📁 Local Root: {LOCAL_ROOT}")
    log(f"🌐 GitHub Repo: {GITHUB_REPO_PATH}")
    log(f"🔄 Git-Push: {'aktiviert' if ENABLE_GIT_PUSH else 'deaktiviert'}")
    
    while True:
        try:
            run_update_cycle()
            
            # Warte 1 Stunde (3600 Sekunden)
            log("😴 Warte 1 Stunde bis zum nächsten Update...")
            time.sleep(3600)
            
        except KeyboardInterrupt:
            log("⏹️ Programm beendet durch Benutzer")
            break
        except Exception as e:
            log(f"❌ Unerwarteter Fehler: {e}")
            log("⏳ Warte 5 Minuten vor erneutem Versuch...")
            time.sleep(300)

if __name__ == "__main__":

    main()