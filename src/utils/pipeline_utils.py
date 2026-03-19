import subprocess
import sys
import os

def get_project_root():
    """Returns the absolute path to the project root."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
def run_script(script_path):

    print(f"\nRunning: {script_path}\n")

    try:
        # Prepare environment to include PROJECT_ROOT in PYTHONPATH
        env = os.environ.copy()
        project_root = get_project_root()
        if "PYTHONPATH" in env:
            env["PYTHONPATH"] = f"{project_root}:{env['PYTHONPATH']}"
        else:
            env["PYTHONPATH"] = project_root

        # Use sys.executable to ensure we use the same Python environment
        subprocess.run([sys.executable, script_path], check=True, env=env)
        print(f"Finished: {script_path}")
        return True
    except subprocess.CalledProcessError:
        print(f"Error running: {script_path}")
        return False