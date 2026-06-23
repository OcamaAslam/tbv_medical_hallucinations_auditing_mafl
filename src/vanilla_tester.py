import logging
from openai import OpenAI
from agentic_tester import setup_logging, NVIDIA_API_KEY

class VanillaModelTester:
    def __init__(self, model_name):
        self.model_name = model_name
        self.client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1", 
            api_key=NVIDIA_API_KEY
        )
        self.logger, self.log_file = setup_logging(model_name, "VANILLA")

    def query_model(self, query):
        self.logger.info(f"Vanilla Query: {query}")
        
        system_prompt = (
            "You are a medical expert. For the following MCQ:\n"
            "1. Briefly state why the correct option is correct.\n"
            "2. Briefly state why the other options are incorrect.\n"
            "3. Conclude your response with: 'The correct option is [LETTER].'"
        )
        
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt}, 
                    {"role": "user", "content": query}
                ],
                temperature=0.1,
                max_tokens=1024
            )
            response = completion.choices[0].message.content
            self.logger.info(f"Vanilla Response: {response}")
            return response
        except Exception as e:
            self.logger.error(f"Vanilla Error: {e}")
            return f"ERROR: {str(e)}"