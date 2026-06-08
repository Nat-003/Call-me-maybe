from src.parser import FunctionCalling, FunctionDefinition

def build_prompt(
        function_definitions: list[FunctionDefinition],
        function_calling: FunctionCalling
        ) -> str:
    prompt = "Available Functions: \n"
    for f in function_definitions:
        prompt += f"Name: {f.name} \n"
        prompt += f"Description: {f.description} \n"
        prompt += "Parameter/s: "
        for param_name, param_type in f.parameters.items():
            prompt += f" {param_name}:"
            prompt += f" {param_type.type},"
        prompt += "\n"
        prompt += f"Returns: {f.returns.type}\n"
        prompt += "\n"
    prompt += "User request: \n"
    prompt += function_calling.prompt
    return prompt