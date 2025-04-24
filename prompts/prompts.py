PROMPT_TOPIC = """
You are a helpful assistant. Your task is to read the given text and identify its main topic. The topic should be a broad category as a nominal phrase (e.g. 'Gay Marriage', 'Climate Change', 'Gun Control', etc.).

Return the result in the following JSON format:

{{
  "article_id": "{unique_id}", # Use this exact ID in your response
  "topic": "<Insert topic here>" # The main topic of the text
}}

Only return the JSON. Do not include any explanation.
"""

PROMPT_STANCE = """
You are a helpful assistant. Your task is to read the given text and identify the main stance towards the topic. The stance should be one of the following three categories: 'Pro', 'Con', or 'Neutral'.

Return the result in the following JSON format:

{{
  "article_id": "{unique_id}", # Use this exact ID in your response
  "topic": "<Insert topic here>" # The main topic of the text
}}

Only return the JSON. Do not include any explanation.
"""