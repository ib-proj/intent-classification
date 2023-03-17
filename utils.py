import re


def clean_swda_utterance(text):
    text = re.sub(r'\[.*?\]', '', text)
    text = text.strip()
    text = re.sub(r'\(.*?\)', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = text.lower()
    text = ' '.join(text.split())
    return {"text": text}


def map_speaker_to_int(data):
    """
    This function maps the speaker information in an example to 0 or 1.
    """
    # Get the caller information from the example
    caller = data["speaker"]

    # Map the caller to 0 or 1
    if caller == "A":
        data["speaker"] = 0
    elif caller == "B":
        data["speaker"] = 1

    return data
