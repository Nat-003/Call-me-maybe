from enum import Enum


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

    def _test(self):
        bug = self._compute_tier2_tokens(self.encoded_name)
        print(self.debug("true"))
        print(self.debug("false"))
        print(self.model.encode('"true"'))
        print(self.model.encode('"false"'))

    def _get_valid_tokens_for_function():
        pass

    def _mask_logits(logits: int, valid_tokens: dict[State, set[int]]):
        pass

    def generate(self, prompt) -> dict:
        # Tier 3 computed here (per prompt, after function chosen)
        # the generation loop
        pass

