import os
import sys
from typing import Callable
from argparse import ArgumentParser
import importlib

from dataset import process_documents_and_store_to_filesystem
from utils import init


def parse_function_string(function_str: str) -> Callable[[dict], str]:
    str_parts = function_str.split(".")
    module_names, function_name = (
        str_parts[:-1], str_parts[-1]
    )
    if len(module_names) == 0:
        # "func"
        return globals()[function_name]

    certain_module_names, maybe_class_name = module_names[:-1], module_names[-1]
    if len(certain_module_names) == 0:
        # "(mod|clz).func"
        module = globals().get(maybe_class_name, None)
        if module is None:
            module = importlib.import_module(maybe_class_name)
        return getattr(module, function_name)

    # (mod)+.(mod|clz).func
    module = importlib.import_module(".".join(certain_module_names))
    if hasattr(module, maybe_class_name):
        # (mod)+.clz.func
        clz = getattr(module, maybe_class_name)
        return getattr(clz, function_name)
    else:
        # (mod)+.mod.func
        module = importlib.import_module(".".join(
            certain_module_names + [maybe_class_name]
        ))
        return getattr(module, function_name)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument('-c', '--collection', type=str, default="datasets", help='ChromaDB collection to be processed')
    parser.add_argument('-o', '--outdir', type=str, default="temp/texts", help='Path to the output directory we want to store processed objects in')
    parser.add_argument(
        '-f', '--function', type=str, default="preprocess.text_operations.ConvertJsonToString.extract_relevant_info",
        help='Path to a Python function we want to use to process objects and compute textual representations.' +
        'The function adheres to the following signature: def func(obj: dict) -> str'
    )
    args = parser.parse_args()
    extract_func = parse_function_string(args.function)

    client = init()
    process_documents_and_store_to_filesystem(
        client,
        collection_name=args.collection,
        savedir=args.outdir,
        extraction_function=extract_func
    )
