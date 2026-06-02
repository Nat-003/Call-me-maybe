from pydantic import BaseModel, Field, ValidationError
import json

class ParameterType(BaseModel):
    type: str



class FunctionCalling(BaseModel):
    prompt: str


class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: dict[str, ParameterType]
    returns: ParameterType


def get_function_definition(file_path: str) -> list[FunctionDefinition] | None:
    try:
        with open(file_path, "r") as file:
            try:
                content = json.load(file)
                if not isinstance(content ,list):
                    print(f"Invalid format in {file_path}: expected a JSON array")
                    return None
                result = []
                for func_def in content:
                    try:
                        result.append(FunctionDefinition.model_validate(func_def))
                    except ValidationError:
                        print("an error occured during Pydantic validation: exiting...")
                        return None
            except json.JSONDecodeError:
                print("Invalid json file")
                return None
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    return result


def get_function_calling(file_path: str) -> list[FunctionCalling]:
    try:
        with open(file_path, "r") as file:
            try:
                content = json.load(file)
                if not isinstance(content, list):
                    print(f"Invalid format in {file_path}: expected a JSON array")
                    return None
                result = []
                for prompt in content:
                    try:
                        result.append(FunctionCalling.model_validate(prompt))
                    except ValidationError:
                        print("an error occured during Pydantic validation: exiting...")
                        return None
            except json.JSONDecodeError:
                print("Invalid json file")
                return None
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None 
    return result      

get_function_definition("data/input/functions_definition.json")
get_function_calling("data/input/function_calling_tests.json")