import os
import sys
import subprocess
import tempfile
import multiprocessing
from typing import Optional

class PythonREPL:
    """A minimal Python REPL allowing multiple-line code execution with working directory support."""

    def __init__(self, working_dir: str = "."):
        # Store and ensure the working directory exists
        self._working_dir = os.path.abspath(working_dir)
        os.makedirs(self._working_dir, exist_ok=True)

    def _run_in_subprocess(self, code: str, queue: multiprocessing.Queue):
        """Writes snippet to a temporary file, runs it in a subprocess with cwd=self._working_dir, and captures output."""
        snippet_file = None
        try:
            # Create a temporary Python file in the specified working directory
            with tempfile.NamedTemporaryFile(
                suffix=".py", prefix="snippet_", dir=self._working_dir, delete=False
            ) as f:
                snippet_file = f.name
                f.write(code.encode("utf-8"))

            # Run the snippet using the current Python interpreter in the desired working directory
            process = subprocess.Popen(
                [sys.executable, snippet_file],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=self._working_dir  # <--- important line
            )
            stdout, stderr = process.communicate()

            # Combine exit code, stdout, and stderr into one string
            combined_output = (
                f"(exit code {process.returncode})\n"
                "==== stdout ====\n"
                f"{stdout}\n"
                "==== stderr ====\n"
                f"{stderr}"
            )
            queue.put(combined_output)

        except Exception as e:
            # Put the exception itself in the queue
            queue.put(repr(e))
        finally:
            # Clean up the temporary file
            if snippet_file and os.path.exists(snippet_file):
                os.remove(snippet_file)

    def run(self, code: str, timeout: Optional[int] = None) -> str:
        """Runs Python code with an optional timeout."""
        result_queue = multiprocessing.Queue()
        process = multiprocessing.Process(
            target=self._run_in_subprocess,
            args=(code, result_queue)
        )
        process.start()
        process.join(timeout)

        if process.is_alive():
            process.terminate()
            return "Execution timed out"

        return result_queue.get()
    

if __name__ == "__main__":
    # Example usage
    repl = PythonREPL()
    code = None
    result = repl.run(code, timeout=5)
    print(result)