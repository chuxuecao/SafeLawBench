import os
import http.client
import json



os.environ["OPENAI_API_KEY"] = "sk-DT3La6LVh0SGc9HtFmvposV6RdsWgjUKtpWAzmJ42mdDOOou"

# os.environ["OPENAI_API_KEY"] = 'sk-ncT0QLumN0zu6vSo3mnt9k2t46Oh6V4m8QZy6NGDD1kyifTt'

os.environ["OPENAI_BASE_URL"] = "https://api3.xhub.chat/v1/"

# https://api3.xhub.chat/v1/chat/completions

from openai import OpenAI
client = OpenAI()

def get_response(system_prompt, user_prompt, model='gpt-4o', temperature=0.0):

   
   model_name = model
   
   response = client.chat.completions.create(
        messages=[{
                     "role": "system",
                     "content": system_prompt
                     },
                     {
                     "role": "user",
                     "content": user_prompt
                     }],
        model= model_name,
        temperature = temperature,
    )

   answer = response.choices[0].message.content
   return answer
# print(get_response('hi', 'hi'))