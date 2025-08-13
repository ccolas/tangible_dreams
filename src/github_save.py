import subprocess
import asyncio
import os

repo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/'

async def save_and_push(cppn, output_path):
    ts = cppn.timestamp
    cppn.save_state()  # saves .pkl and .png into output_path
    rel_output_path = os.path.relpath(output_path, repo_path)
    try:
        subprocess.run(["git", "add", rel_output_path], cwd=repo_path, check=True)
        # Commit with timestamp in message
        commit_msg = f"Add outputs for {ts}"
        subprocess.run(["git", "commit", "-m", commit_msg], cwd=repo_path, check=True)
        subprocess.run(["git", "push"], cwd=repo_path, check=True)
        print(f"[INFO] Saved and pushed outputs for {ts}")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Git command failed: {e}")

