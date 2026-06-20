
from src.parser import get_function_calling, get_function_definition, vocab_loader
from src.promptbuilder import build_prompt
import argparse
from llm_sdk import Small_LLM_Model
from src.decoder import Decoder
import json


def main() -> None:
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--functions_definition", type=str, default="data/input/functions_definition.json")
        parser.add_argument("--input", type=str, default="data/input/function_calling_tests.json")
        # parser.add_argument("--output", type=str)
        args = parser.parse_args()         
        # args.output
        function_definition =  get_function_definition(args.functions_definition)
        function_calling = get_function_calling(args.input )
        if function_definition is None or function_calling is None:
            return
        model = Small_LLM_Model()
        vocab = vocab_loader(model)
        decoder = Decoder(model, vocab, function_definition)
        
        for prompt in function_calling:
            prompt_f = build_prompt(function_definition, prompt)
            result = decoder.generate(prompt_f)
            print(result)
    except FileNotFoundError:
        print("Invalide fiel path")
if __name__ == "__main__":
    main()


