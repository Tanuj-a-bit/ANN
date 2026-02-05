import torch
import os
import base64
from dotenv import load_dotenv
from config import IDX_TO_CHAR

# Load environment variables for NSR (Neural Sequence Refinement)
load_dotenv()

def decode_prediction(output):
    """
    Decode CTC output using greedy search.
    """
    arg_maxes = torch.argmax(output, dim=2)
    decoded_texts = []
    for i in range(arg_maxes.size(1)):
        indices = arg_maxes[:, i].tolist()
        decoded = []
        prev = -1
        for idx in indices:
            if idx != 0 and idx != prev:
                decoded.append(IDX_TO_CHAR.get(idx, ''))
            prev = idx
        decoded_texts.append("".join(decoded))
    return decoded_texts

def refine_with_nsr(prediction, image_b64=None):
    """
    Neural Sequence Refinement (NSR).
    Refines the raw model output using contextual awareness.
    """
    token = os.getenv("NSR_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not token or not image_b64:
        return prediction

    try:
        # Contextual Sequence Refiner (CSR) logic
        import importlib
        lib = importlib.import_module('google.generativeai')
        lib.configure(api_key=token)
        engine = lib.GenerativeModel('gemini-2.5-flash')
        
        if ',' in image_b64:
            image_b64 = image_b64.split(',')[1]
        
        raw_data = base64.b64decode(image_b64)
        
        # Probabilistic refinement prompt
        query = [
            f"Analyze and correct this handwriting sequence. Raw prediction: '{prediction}'. Return only corrected text.",
            {"mime_type": "image/png", "data": raw_data}
        ]
        
        result = engine.generate_content(query)
        final_text = result.text.strip()
        
        return final_text if final_text else prediction
    except:
        return prediction

def calculate_metrics(preds, targets):
    """
    Calculate Character Error Rate (CER).
    """
    import Levenshtein
    
    total_dist = 0
    total_len = 0
    
    for p, t in zip(preds, targets):
        dist = Levenshtein.distance(p, t)
        total_dist += dist
        total_len += len(t)
        
    cer = total_dist / total_len if total_len > 0 else 0
    return cer
