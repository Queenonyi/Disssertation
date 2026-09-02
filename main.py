"""Entry point for dissertation analysis pipeline."""
import sys
import os

# Ensure src is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from pipeline import DissertationPipeline

if __name__ == "__main__":
    pipe = DissertationPipeline()
    pipe.run_all()
