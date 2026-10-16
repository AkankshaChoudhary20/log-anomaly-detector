import subprocess
import sys
from pathlib import Path


def test_first_run_creates_output_directory(tmp_path):
    script = Path(__file__).resolve().parents[1] / 'src/main.py'
    result = subprocess.run([sys.executable, str(script)], cwd=tmp_path,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / 'output/alerts.log').is_file()
