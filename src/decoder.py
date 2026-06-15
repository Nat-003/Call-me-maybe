from enum import Enum
from typing import Any
import numpy as np

class State(Enum):
    EXPECTING_OPEN_BRACE = "expecting_open_brace"
    EXPECTING_NAME = "expecting_name"
    EXPECTING_COLON = "expecting_colon"
    INSIDE_FUNCTION_NAME = "inside_function_name"
    EXPECTING_COMMA = "expecting_comma"
    EXPECTING_PARAMETERS = "expecting_parameters"
    EXPECTING_OPEN_PARAMETER_BRACE = "expecting_open_parameter_brace"
    EXPECTING_PARAMETER_KEY = "expecting_parameter_key"
    INSIDE_PARAMETER_VALUE_NUMBER = "inside_parameter_value_number"
    INSIDE_PARAMETER_VALUE_STRING = "inside_parameter_value_string"
    INSIDE_PARAMETER_VALUE_BOOLEAN = "inside_parameter_value_boolean"
    EXPECTING_CLOSING_PARAMETER_BRACE = "expecting_closing_parameter_brace"
    EXPECTING_CLOSING_BRACE = "expecting_closing_brace"
    DONE = "done"


class Decoder:
    def __init__(self, model, vocab, function_definitions):
        self.model = model
        self.vocab = vocab
        self.function_definitions = function_definitions
        self.current_state = State.EXPECTING_OPEN_BRACE
        self.valid_tokens_per_state = self._precompute_valid_tokens()
        self.generate_within_state = ""
        self.position_within_state = 0
        self.encoded_name = self.model.encode('"name"').tolist()[0]
        self.encoded_parameters = self.model.encode('"parameters"').tolist()[0]
        self.encoded_function_names = {
                                        func.name: self.model.encode('"' + func.name + '"').tolist()[0]
                                        for func in self.function_definitions
                                        }
        self.generated_tokens_within_state = []

    def _get_tokens_for_string(self, target: str) -> set[int]:
        result = set()
        for token_id, token_str in self.vocab.items():
            if token_str == target:
                result.add(token_id)
        return result

    def debug(self, target):
        for token_id, token_str in self.vocab.items():
            if target in token_str:
                print(f"{token_id} -> {token_str}")
    
    def _precompute_valid_tokens(self) -> dict[State, set[int]]:
        result = {}
        result[State.EXPECTING_OPEN_BRACE] = self._get_tokens_for_string("{")
        result[State.EXPECTING_COLON] = self._get_tokens_for_string(":")
        result[State.EXPECTING_COMMA] = self._get_tokens_for_string(",")
        result[State.EXPECTING_OPEN_PARAMETER_BRACE] = self._get_tokens_for_string("{")
        result[State.EXPECTING_CLOSING_PARAMETER_BRACE] = self._get_tokens_for_string("}")
        result[State.EXPECTING_CLOSING_BRACE] = self._get_tokens_for_string("}")
        result[State.INSIDE_FUNCTION_NAME] = self._inside_function_name()
        result[State.INSIDE_PARAMETER_VALUE_NUMBER] = self._get_number_tokens()
        result[State.INSIDE_PARAMETER_VALUE_BOOLEAN] = self._get_boolean_tokens()
        result[State.INSIDE_PARAMETER_VALUE_STRING] = self._get_string_tokens()
        result[State.EXPECTING_PARAMETER_KEY] = self._expecting_paramerter_key()
        return result

    def _compute_tier2_tokens(self, target: list[int]) -> set[int]:
        result = set()
        if self.position_within_state >= len(target):
            return result
        else:
            result.add(target[self.position_within_state])
        return result
    
    def _inside_function_name(self) -> set[int]:
        result = set()
        for func in self.function_definitions:
            func_name = self.model.encode('"' + func.name + '"').tolist()[0]
            for id in func_name:
                result.add(id)
        return result

    def _expecting_paramerter_key(self) -> set[int]:
        result = set()
        for func in self.function_definitions:
            params = func.parameters.keys()
            for k in params:
                func_params = self.model.encode('"' + k + '"').tolist()[0]
                for key_params in func_params:
                    result.add(key_params)
        return result

    def _get_number_tokens(self) -> set[int]:
        result = set()
        for token_id, token_str in self.vocab.items():
            if token_str != "" and all(c.isdigit() or c == '.' or  c == '-' for c in token_str):
                result.add(token_id)
        return result

    def _get_boolean_tokens(self) -> set[int]:
        result = set()
        for token_id, token_str in self.vocab.items():
            if token_str == "true" or token_str == "false":
                result.add(token_id)
        return result

    def _get_string_tokens(self) -> set[int]:
        result = set()
        for token_id, token_str in self.vocab.items():
            if '"' not in token_str or token_str == '"':
                result.add(token_id)
        return result

    def _test(self):
        bug = self._compute_tier2_tokens(self.encoded_name)
        print(self.debug("true"))
        print(self.debug("false"))
        print(self.model.encode('"true"'))
        print(self.model.encode('"false"'))

    def _get_valid_tokens_for_function():
        pass

    def _mask_logits(self ,logits: Any, valid_tokens: set[int]) -> Any:
        arr = logits.detach().numpy()
        cpy = arr.copy()
        arr[:] = -float('inf')
        arr[list(valid_tokens)] = cpy[list(valid_tokens)]
        return arr

    def generate(self, prompt) -> dict:
        immediate_transitions = {
    State.EXPECTING_OPEN_BRACE: State.EXPECTING_NAME,
    State.EXPECTING_NAME: State.EXPECTING_COLON,
    State.EXPECTING_PARAMETERS: State.EXPECTING_COLON,
    State.EXPECTING_OPEN_PARAMETER_BRACE: State.EXPECTING_PARAMETER_KEY,
    State.EXPECTING_CLOSING_PARAMETER_BRACE: State.EXPECTING_CLOSING_BRACE,
    State.EXPECTING_CLOSING_BRACE: State.DONE,
}
        input_ids = self.model.encode(prompt)
        result = {"prompt": prompt, "name": "", "parameters": {}}
        previous_state = None
        while self.current_state != State.DONE:
            valid_tokens = self.valid_tokens_per_state[self.current_state]
            logits = self.model.get_logits_from_input_ids(input_ids)
            clean_tokens = self._mask_logits(logits, valid_tokens)
            highest_token_score = np.argmax(clean_tokens)
            input_ids.append(highest_token_score)
            
            if self.current_state == State.EXPECTING_COLON:
                if previous_state == State.EXPECTING_NAME:
                    self.current_state = State.INSIDE_FUNCTION_NAME
                elif previous_state == State.EXPECTING_PARAMETERS:
                    self.current_state = State.EXPECTING_OPEN_PARAMETER_BRACE
            elif self.current_state in immediate_transitions:
                self.current_state = immediate_transitions[self.current_state]
            elif self.current_state == State.INSIDE_FUNCTION_NAME:
                self.generated_tokens_within_state.append(highest_token_score)
                if self.generated_tokens_within_state in self.encoded_function_names.values():
                    self.current_state = State.EXPECTING_COMMA
                    for name, tokens in self.encoded_function_names.items():
                        if tokens == self.generated_tokens_within_state:
                            result["name"] = name
                    self.generated_tokens_within_state = []  # reset for next state
            elif self.current_state == State.EXPECTING_COMMA:
                if previous_state == State.INSIDE_FUNCTION_NAME:
                    self.current_state = State.EXPECTING_PARAMETERS
                elif previous_state in {
                    State.INSIDE_PARAMETER_VALUE_NUMBER,
                    State.INSIDE_PARAMETER_VALUE_STRING,
                    State.INSIDE_PARAMETER_VALUE_BOOLEAN
                }:
                    self.current_state = State.EXPECTING_PARAMETER_KEY
            previous_state = self.current_state
    

