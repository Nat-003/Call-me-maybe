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
        # Tier 1 + Tier 2 precomputed here (once)
        self.valid_tokens_per_state = self._precompute_valid_tokens()
    
    def _get_tokens_for_string(self, target: str) -> set[int]:
        result = set()
        for token_id, token_str in self.vocab.items():
            if token_str == target:
                result.add(token_id)
        return result

    def _precompute_valid_tokens(self) -> dict[State, set[int]]:
        result = {}
        result[State.EXPECTING_OPEN_BRACE] = self._get_tokens_for_string("{")
        return result

    def generate(self, prompt) -> dict:
        # Tier 3 computed here (per prompt, after function chosen)
        # the generation loop
        pass
