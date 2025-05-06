import json
import openai
from tqdm import tqdm
from config import API_KEY, BASE_URL, MODEL

# Initialize OpenAI client
# client = openai.OpenAI(api_key=API_KEY)
client = openai.OpenAI(api_key=API_KEY, base_url=BASE_URL, max_retries=20)

def load_dataset(file_path="dataset.json"):
    with open(file_path, 'r') as f:
        return json.load(f)
    

def generate_batch_json():
    dataset = load_dataset()
    import json

    with open("batch_requests.jsonl", "w") as f:
        for idx, entry in enumerate(dataset, start=1):
            text = entry["string_cmd"]
            request = {
                "custom_id": f"request-{idx}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": MODEL,
                    "messages": [
                        {
                            "role": "system",
                            "content": "Paraphrase the following instruction while keeping the same meaning and intent. Your answer should only be one sentence, do not give multiple options or extra clarification or text in addition to the paraphrased instruction"
                        },
                        {"role": "user", "content": text}
                    ],
                    "max_tokens": 1500
                }
            }
            f.write(json.dumps(request) + "\n")


def upload_batch_file():
    batch_input_file = client.files.create(
        file=open("batch_requests.jsonl", "rb"),
        purpose="batch"
    )

    print(batch_input_file)

    batch_input_file_id = batch_input_file.id
    batch = client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
            "description": "robollm dataset augmentation fr"
        }
    )

    print(batch)

    batch_retrieved = client.batches.retrieve(batch.id)
    print(batch_retrieved)


def check_batch():
    batch_id = "batch_id_here" 
    batch_retrieved = client.batches.retrieve(batch_id)
    print(batch_retrieved)

    if out := batch_retrieved.output_file_id:
        file_response = client.files.content(out)
        print(file_response)
        file_response.write_to_file('batch_results.jsonl')


def merge_batch_results(results_file="batch_results.jsonl", dataset_file="dataset.json", output_file="dataset_merged.json"):
    """
    Reads the batch results from a JSONL file, matches them to the original dataset entry using the custom_id,
    and adds the paraphrased text as a new field 'paraphrased_cmd'.
    
    The custom_id should be in the format "request-{number}", corresponding to the dataset entry,
    where the dataset was originally enumerated starting at 1.
    """
    import json

    # Load the original dataset
    with open(dataset_file, "r") as f:
        dataset = json.load(f)

    # Read the batch results
    with open(results_file, "r") as f:
        for line in f:
            if not line.strip():
                continue
            result = json.loads(line)
            custom_id = result.get("custom_id", "")
            # Extract the request number from custom_id (e.g. "request-10")
            try:
                request_num = int(custom_id.split("-")[1])
            except (IndexError, ValueError):
                print(f"Skipping invalid custom_id: {custom_id}")
                continue

            # Extract the paraphrased text from the response object
            try:
                paraphrased_text = result["response"]["body"]["choices"][0]["message"]["content"]
            except (KeyError, IndexError):
                print(f"Skipping entry with missing paraphrased text for custom_id: {custom_id}")
                continue

            # Update the corresponding dataset entry (accounting for enumerate starting at 1)
            idx = request_num - 1
            if idx < len(dataset):
                dataset.append({"cmd_type": dataset[idx]['cmd_type'], "string_cmd": paraphrased_text, 'structured_cmd': dataset[idx]['structured_cmd']})
            else:
                print(f"Custom id {custom_id} refers to dataset index {idx} which is out of bounds")

    # Save the merged dataset
    with open(output_file, "w") as f:
        json.dump(dataset, f, indent=2)
    print(f"Merged dataset written to {output_file}")


def replace_on_dataset(original_dataset="dataset_merged.json", take_from="dataset.json", output_dataset="final_dataset.json"):
    """
    Replaces the entries in the original dataset with the entries from the take_from dataset up to a certain index.
    """
    import json

    # Load the original dataset
    with open(original_dataset, "r") as f:
        original_data = json.load(f)

    # Load the dataset to take from
    with open(take_from, "r") as f:
        take_data = json.load(f)

    # Replace entries in the original dataset
    # for i in range(len(take_data)):
    #     if i < len(original_data):
    #         original_data[i] = take_data[i]

    for i, entry in enumerate(take_data):
        if i < len(original_data):
            original_data[i]['string_cmd'] = entry['string_cmd']
            original_data[i]['cmd_type'] = entry['cmd_type']
            original_data[i]['structured_cmd'] = entry['structured_cmd']
        else:
            print(f"Index {i} is out of bounds for the original dataset")
            break


    # Save the updated dataset
    with open(output_dataset, "w") as f:
        json.dump(original_data, f, indent=2)
    print(f"Updated dataset written to {output_dataset}")



def paraphrase_text(text):
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "Paraphrase the following instruction while keeping the same meaning and intent. Your answer should only be one sentence, do not give multiple options or extra clarification or text in addition to the paraphrased instruction"},
                {"role": "user", "content": text}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error paraphrasing: {e}")
        return text

def paraphrase_dataset():
    # Load the dataset
    dataset = load_dataset()
    
    # Calculate the starting index for the second half
    start_idx = len(dataset) // 2
    
    # Process second half of entries
    for i in tqdm(range(start_idx, len(dataset))):
        original_text = dataset[i]['string_cmd']
        # Add delay to respect API rate limits
        # time.sleep(0.5)
        response = paraphrase_text(original_text)
        # Extract text after </think> if present
        if '</think>' in response:
            response = response.split('</think>')[-1].strip()
        dataset[i]['string_cmd'] = response

        print(f"""
        Original: {original_text}
        Paraphrased: {dataset[i]['string_cmd']}
        """)
    
    # Save the updated dataset
    with open('dataset_paraphrased.json', 'w') as f:
        json.dump(dataset, f, indent=2)

if __name__ == "__main__":
    paraphrase_dataset()
    # generate_batch_json()
    # upload_batch_file()
    # check_batch()
    # merge_batch_results()
    # replace_on_dataset()