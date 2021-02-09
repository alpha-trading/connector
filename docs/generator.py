import ast
import json
import sys
from os import path
from docstring_parser import parse


def docs_to_parsed_data(docs):
    tree = ast.parse(docs)
    functions = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        docstring = ast.get_docstring(node)
        if docstring is None:
            continue

        parsed_docstring = parse(docstring)
        # print(docstring)
        params = []
        for param in parsed_docstring.params:
            params.append({"name": param.arg_name, "description": param.description, "is_optional": param.is_optional})
        function_data = {
            "name": node.name,
            "korean_name": parsed_docstring.short_description.replace(" ", ""),
            "params": params,
            "description": parsed_docstring.long_description,
        }
        functions.append(function_data)
    return functions


current_dir = path.dirname(__file__)
with open(path.join(current_dir, "../utils/indicator.py")) as indicator_file:
    functions = docs_to_parsed_data(indicator_file.read())
    with open(path.join(current_dir, "json/indicator.json"), "w") as json_file:
        json.dump(functions, json_file)

with open(path.join(current_dir, "../utils/function.py")) as indicator_file:
    functions = docs_to_parsed_data(indicator_file.read())
    with open(path.join(current_dir, "json/function.json"), "w") as json_file:
        json.dump(functions, json_file)
