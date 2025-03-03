
from openai import OpenAI
import tiktoken
import json
import pandas as pd




encoding = tiktoken.encoding_for_model("gpt-4o-mini")


client = OpenAI(api_key= "") 

def process_response_to_dataframe(response_text):
    responses = response_text.strip().split("\n")
    data = []
    for response in responses:
        json_response = json.loads(response)
        custom_id = json_response.get('custom_id')
        content = json_response.get('response', {}).get('body', {}).get('choices', [{}])[0].get('message', {}).get('content')
        data.append({"custom_id": custom_id, "content": content})
    batch = pd.DataFrame(data)
    return batch

def process_and_save_batch(output_file_id, save_directory):
    file_response = client.files.content(output_file_id)
    response_text = file_response.text

    batch_df = process_response_to_dataframe(response_text)

    batch_df.to_csv(f'{save_directory}/cloze.csv', index=False)
    print(f"saved as cloze.csv")