import ast
import re
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
            korean_name = re.search(r"(?<=\().+?(?=\))", param.description).group()
            description = param.description.replace(f"({korean_name})", "")
            params.append(
                {
                    "name": param.arg_name,
                    "description": description,
                    "korean_name": korean_name,
                    "is_optional": param.is_optional,
                }
            )
        function_data = {
            "name": node.name,
            "korean_name": parsed_docstring.short_description.replace(" ", ""),
            "params": params,
            "description": parsed_docstring.long_description,
        }

        functions.append(function_data)
    return functions


def markdown_generator(function_data):
    params_string = ""
    for param in function_data["params"]:
        params_string += f"`{param['name']}: {param['description']}`\n\n"
    md_string = f"""## {function_data['korean_name']}-{function_data['name']}

{function_data['description']}

```python
{function_data['name']}({', '.join([param['name'] for param in function_data['params']])})
{function_data['korean_name']}({', '.join([param['korean_name'] for param in function_data['params']])})
```
### 변수
{params_string}
"""
    md_string = md_string.replace("\n\n", "  \n")  # 마크다운의 개행에 맞게 변경
    md_string = md_string.replace("<설명>", "")
    md_string = md_string.replace("<사용 방법>", "### 사용 방법")
    md_string = md_string.replace("<계산 방법>", "### 계산 방법")
    md_string = md_string.replace("'", "`")
    return md_string


current_dir = path.dirname(__file__)
with open(path.join(current_dir, "../utils/indicator.py")) as indicator_file:
    functions = docs_to_parsed_data(indicator_file.read())
    with open(path.join(current_dir, "md/indicator.md"), "w") as md_file:
        for function_data in functions:
            md_file.write(markdown_generator(function_data))

with open(path.join(current_dir, "../utils/function.py")) as indicator_file:
    functions = docs_to_parsed_data(indicator_file.read())
    with open(path.join(current_dir, "md/function.md"), "w") as md_file:
        for function_data in functions:
            md_file.write(markdown_generator(function_data))
