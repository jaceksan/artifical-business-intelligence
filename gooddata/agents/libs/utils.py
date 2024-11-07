import os
from functools import wraps
from pathlib import Path
from time import perf_counter

DEBUG_PATH = Path("tmp")
PROMPT_PATH = Path("prompts")


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        final_func_name = func.__name__
        print(f"Function {final_func_name} started")
        start_time = perf_counter()
        result = func(*args, **kwargs)
        end_time = perf_counter()
        total_time = end_time - start_time
        print(f"Function {final_func_name} took {total_time:.4f} seconds")
        return result

    return timeit_wrapper


def debug_to_file(file_name: str, content: str, folder: Path = None) -> None:
    if folder:
        file_path = DEBUG_PATH / folder / file_name
        if not os.path.isdir(DEBUG_PATH / folder):
            os.makedirs(DEBUG_PATH / folder)
    else:
        file_path = DEBUG_PATH / file_name
    with open(file_path, "w") as fp:
        fp.write(content)


def replace_in_string(content: str, replacements: dict) -> str:
    for key, value in replacements.items():
        content = content.replace(f"{{{key}}}", value)
    return content


def prompt_from_file(file_path: Path, replacements: dict = None) -> str:
    with open(PROMPT_PATH / file_path, "r") as fp:
        prompt = fp.read()
        # We cannot use format function, because it changes JSON(dict) examples in the text
        if replacements:
            return replace_in_string(prompt, replacements)
        else:
            return prompt
