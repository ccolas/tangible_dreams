import subprocess
import os
from pathlib import Path
from  datetime import datetime

repo_path = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/')

def write_readme(output_path):
    # Collect *.png and sort newest→oldest
    names = sorted((p.name for p in output_path.glob("*.png")), reverse=True)

    by_day = {}
    for name in names:
        # Expect filenames like image_YYYY_MM_DD_HHMMSS.png
        parts = name.split("_")
        if len(parts) < 4:
            continue
        try:
            y, m, d = map(int, parts[1:4])
        except ValueError:
            continue
        day_obj = datetime(y, m, d).date()
        by_day.setdefault(day_obj, []).append(name)

    lines = ["# Tangible Dreams\n\nHere are the patterns shaped by my friends and I.\n\n"]
    for day_obj in sorted(by_day.keys(), reverse=True):
        day_str = day_obj.strftime("%B %d, %Y")
        lines.append(f"## {day_str}\n")
        lines.append("<p style='display:flex;flex-wrap:wrap;gap:8px'>")
        for name in by_day[day_obj]:
            lines.append(f'<img src="./{name}" alt="{name}" width="300" loading="lazy" />')
        lines.append("</p>\n")

    readme_path = output_path / "README.md"
    readme_path.write_text("\n".join(lines), encoding="utf-8")

    return readme_path

async def save_and_push(cppn):
    print('[SAVE] saving current network')
    ts = cppn.timestamp
    cppn.save_state()  # saves .pkl and .png into output_path
    readme_path = write_readme(cppn.output_path)
    rel_output_path = os.path.relpath(cppn.output_path, repo_path)
    rel_readme_path = os.path.relpath(readme_path, repo_path)
    try:
        subprocess.run(["git", "add", rel_output_path, rel_readme_path], cwd=repo_path, check=True)
        # Commit with timestamp in message
        commit_msg = f"Add outputs for {ts}"
        subprocess.run(["git", "commit", "-m", commit_msg], cwd=repo_path, check=True)
        subprocess.run(["git", "push"], cwd=repo_path, check=True)
        print(f"[INFO] Saved and pushed outputs for {ts}")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Git command failed: {e}")

