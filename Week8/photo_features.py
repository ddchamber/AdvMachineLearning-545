import os
import csv
import json
import torch
import time
import boto3
import pandas as pd
from PIL import Image
from dotenv import load_dotenv
from transformers import BlipProcessor, BlipForConditionalGeneration

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ---------- LOAD ENV ----------
load_dotenv("/Users/dan/calpoly/BusinessAnalytics/GSB570GENAI/my_discord_bot/.env")
aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
aws_region = os.getenv("AWS_DEFAULT_REGION")

# ---------- SETTINGS ----------
image_folder = "/Users/dan/calpoly/BusinessAnalytics/GSB545ADML/Week8/Alex_Kelly_Pics/TestSet"
output_csv = "/Users/dan/calpoly/BusinessAnalytics/GSB545ADML/Week8/Alex_Kelly_Pics/testset_llm_annotations.csv"
features = [
    "human", "castle", "indoors", "city", "suburb", "rural", "trees", "night",
    "mountain", "water", "architecture", "food",
    "animals", "sky_prominent", "symmetry", "colorful", "vehicle", "foreground_focus"
]
model_id = "anthropic.claude-3-sonnet-20240229-v1:0"

# ---------- LOAD BLIP ----------
print("Loading BLIP captioning model...")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# ---------- SETUP BEDROCK ----------
print("Setting up AWS Bedrock client...")
bedrock = boto3.client(
    "bedrock-runtime",
    region_name=aws_region,
    aws_access_key_id=aws_access_key,
    aws_secret_access_key=aws_secret_key
)

# ---------- UTILITIES ----------
def generate_caption(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(image, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=50)  # more descriptive
    return processor.decode(out[0], skip_special_tokens=True)

def retry_invoke_model(**kwargs):
    for attempt in range(3):
        try:
            return bedrock.invoke_model(**kwargs)
        except botocore.exceptions.ClientError as e:
            if "ThrottlingException" in str(e):
                wait = 2 ** attempt
                print(f"⏳ Throttled. Waiting {wait}s...")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("Too many throttling errors.")

import re  # place at top with other imports if not already there

def extract_feature_list(text_output, expected_length=18):
    """
    Extracts the first valid JSON-style list of 0s and 1s from LLM response.
    """
    match = re.search(r"\[(?:\s*\d\s*,)*\s*\d\s*\]", text_output)
    if not match:
        print("⚠️ No list found in LLM response:")
        print(text_output)
        return [None] * expected_length

    try:
        result = json.loads(match.group(0))
    except json.JSONDecodeError:
        print("⚠️ Failed to decode list:")
        print(match.group(0))
        return [None] * expected_length

    if len(result) != expected_length:
        print(f"⚠️ Parsed list has wrong length ({len(result)} != {expected_length}):")
        print(result)
        return [None] * expected_length

    return result

def classify_with_bedrock(caption):
    messages = [
        {
            "role": "user",
            "content": f"""
You are helping annotate images with binary features. Below is a detailed caption:

"{caption}"

Label each feature with '1' (present) or '0' (not present).

Features:
- human
- castle
- indoors
- city
- suburb
- rural
- trees
- night
- mountain
- water
- architecture
- food
- animals
- sky_prominent
- symmetry
- colorful
- vehicle
- foreground_focus

Output the result in this format:
[1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1]
"""
        }
    ]

    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "messages": messages,
        "max_tokens": 100,
        "temperature": 0.2
    })

    response = retry_invoke_model(
        modelId=model_id,
        body=body,
        contentType="application/json",
        accept="application/json"
    )

    response_body = json.loads(response['body'].read())
    content_blocks = response_body.get("content", [])
    if isinstance(content_blocks, list) and content_blocks:
        text_output = content_blocks[0].get("text", "").strip()
    else:
        text_output = ""

    result = extract_feature_list(text_output, expected_length=len(features))
    return result

# ---------- LOAD EXISTING ----------
if os.path.exists(output_csv):
    existing_df = pd.read_csv(output_csv)
    processed_images = set(existing_df['image'].tolist())
    print(f"Resuming... {len(processed_images)} already done.")
else:
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["image", "caption", "raw_response"] + features)
    processed_images = set()

# ---------- MAIN LOOP ----------
print("🔍 Starting image annotation...")
for filename in sorted(os.listdir(image_folder)):
    if not filename.endswith(".png") or filename in processed_images:
        continue

    try:
        path = os.path.join(image_folder, filename)
        print(f"\nProcessing {filename}...")

        caption = generate_caption(path)
        print(f"  → Caption: {caption}")

        labels = classify_with_bedrock(caption)
        print(f"  → Features: {labels}")

        # Save to CSV immediately
        with open(output_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([filename, caption, json.dumps(labels)] + labels)

        time.sleep(0.5)  # anti-throttle pause

    except Exception as e:
        print(f"Error on {filename}: {e}")

print(f"\nFinished! All annotations saved to {output_csv}")