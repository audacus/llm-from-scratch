import urllib.request
from pathlib import Path


def get_the_verdict() -> str:
    file_path = "the-verdict.txt"

    if not Path(file_path).is_file():
        # Load and save `The Verdict` by Edith Wharton to a file.
        url = ("https://raw.githubusercontent.com/rasbt/"
               "LLMs-from-scratch/refs/heads/main/ch02/01_main-chapter-code/"
               "the-verdict.txt")
        urllib.request.urlretrieve(url, file_path)

    # Load the file.
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    return text
