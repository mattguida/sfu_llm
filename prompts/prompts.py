PROMPT_TOPIC_STANCE = """
You are a helpful assistant. Your task is to read the given {content_type} and:
1. Identify its main topic as a broad category in a nominal phrase (e.g. "Climate Change", "Gun Control", "Immigration", etc.).
2. Identify the stance taken in the {content_type} **on that topic**. The stance should be one of:
   - "Pro"
   - "Against"
   - "Neutral"

Return the result in the following JSON format:

{{
  "{content_type}_id": "{unique_id}", 
  "topic": "<Insert topic here>",
  "stance": "<Insert stance here>" 
}}

Only return the JSON. Do not include any explanation.
"""
