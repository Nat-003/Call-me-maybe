from src.parser import get_function_calling, get_function_definition, vocab_loader
from src.promptbuilder import build_prompt
import argparse
from llm_sdk import Small_LLM_Model
import json


def main() -> None:
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--functions_definition", type=str, default="data/input/functions_definition.json")
        parser.add_argument("--input", type=str, default="data/input/function_calling_tests.json")
        # parser.add_argument("--output", type=str)
        args = parser.parse_args()         
        # args.output
        result =  get_function_definition(args.functions_definition)
        result2 = get_function_calling(args.input )
        if result is None or result2 is None:
            return
        model = Small_LLM_Model()
        for prompt in result2:
            built_prompt = build_prompt(result, prompt)
        
        print(vocab_loader(model))
            
    except FileNotFoundError:
        print("Invalide fiel path")
if __name__ == "__main__":
    main()


