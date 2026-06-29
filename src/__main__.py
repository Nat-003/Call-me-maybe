
from src.parser import get_function_calling, get_function_definition, vocab_loader
from src.promptbuilder import build_prompt
import argparse
from llm_sdk import Small_LLM_Model
from src.decoder import Decoder
from src.output import generate_output
import json


def main() -> None:
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--functions_definition", type=str, default="data/input/functions_definition.json")
        parser.add_argument("--input", type=str, default="data/input/function_calling_tests.json")
        parser.add_argument("--output", type=str)
        args = parser.parse_args()         
        args.output
        function_definition =  get_function_definition(args.functions_definition)
        function_calling = get_function_calling(args.input )
        if function_definition is None or function_calling is None:
            return
        model = Small_LLM_Model()
        vocab = vocab_loader(model)
        decoder = Decoder(model, vocab, function_definition)
        data = []
        for item in function_calling:
            prompt_f = build_prompt(function_definition, item)
            result = decoder.generate(prompt_f)
            result["prompt"] = item.prompt
            print(item.prompt)   # overwrite enriched → original
            data.append(result)
        generate_output(data, args.output)
        # result = decoder._string_valid_parameter()
        # print(result)
    except FileNotFoundError:
        print("Invalide fiel path")
if __name__ == "__main__":
    main()


