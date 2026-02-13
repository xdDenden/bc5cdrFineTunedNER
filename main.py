import re
import random
import spacy
from spacy import displacy
from pathlib import Path
import pandas as pd
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    pipeline
)
from tqdm import tqdm

# ==========================================
# CONFIGURATION
# ==========================================
DATASET_NAME = "tner/bc5cdr"
YOUR_MODEL_PATH = "./final_clinical_ner_model"
PRETRAINED_MODEL = "en_ner_bc5cdr_md"

# Using 50% of test set for stats, visualizing 10
DATASET_PERCENTAGE = 0.5
VISUALIZATION_COUNT = 10

# Colors for visualization
COLORS = {
    "CHEMICAL": "linear-gradient(90deg, #aa9cfc, #fc9ce7)",
    "DISEASE": "linear-gradient(90deg, #ff9a8d, #ff6961)",
    "DOSAGE": "linear-gradient(90deg, #feca57, #ff9ff3)",
}


# ==========================================
# HELPER FUNCTIONS
# ==========================================

def get_dataset_labels(dataset):
    """
    Dynamically attempts to get labels.
    If failing, falls back to the CORRECT TNER mapping.
    """
    try:
        features = dataset['test'].features
        if 'tags' in features:
            label_list = features['tags'].feature.names
            print(f"   [INFO] Detected Dataset Labels: {label_list}")
            return {i: label for i, label in enumerate(label_list)}
    except Exception as e:
        print(f"   [WARN] Could not detect labels dynamically ({e}). Using TNER fallback.")

    # CORRECTED TNER MAPPING (B tags usually come first)
    return {
        0: "O",
        1: "B-CHEMICAL",
        2: "B-DISEASE",  # Previously this was I-CHEMICAL (Wrong!)
        3: "I-CHEMICAL",
        4: "I-DISEASE"
    }


def extract_dosages(text):
    """Regex for Dosage"""
    dosage_pattern = r'\b\d+\.?\d*\s?(mg|g|ml|mcg|units?)\b'
    dosages = []
    for match in re.finditer(dosage_pattern, text, re.IGNORECASE):
        dosages.append({
            "start": match.start(),
            "end": match.end(),
            "label": "DOSAGE",
            "text": match.group()
        })
    return dosages


def reconstruct_and_get_gt(tokens, tags, id2label):
    """
    Reconstructs text and extracts Ground Truth.
    INCLUDES FIX FOR ORPHAN I- TAGS.
    """
    text = ""
    entities = []
    current_ent = None

    for token, tag_id in zip(tokens, tags):
        start = len(text)
        text += token + " "
        end = len(text) - 1

        label_full = id2label.get(tag_id, "O")

        # 1. Check if it's a Start Tag (B-) OR an orphan Inside Tag (I- without predecessor)
        is_start = label_full.startswith("B-")
        is_continue = label_full.startswith("I-")

        # If we see an I- tag but have no current entity, treat it as a B- tag (Auto-Fix)
        if is_continue and current_ent is None:
            is_start = True
            is_continue = False

        # If we see an I- tag but the type switched (e.g. Chemical -> Disease), treat as new B-
        if is_continue and current_ent and label_full[2:] != current_ent['label']:
            # Close old
            entities.append(current_ent)
            current_ent = None
            # Start new
            is_start = True
            is_continue = False

        if is_start:
            if current_ent: entities.append(current_ent)
            current_ent = {
                "start": start, "end": end, "label": label_full[2:].upper(), "text": token
            }
        elif is_continue and current_ent:
            current_ent["end"] = end
            current_ent["text"] += " " + token
        else:
            # It's 'O' or a weird break
            if current_ent:
                entities.append(current_ent)
                current_ent = None

    if current_ent: entities.append(current_ent)
    return text.strip(), entities


def get_model_predictions(text, pipe):
    """Your Custom BioBERT Predictions"""
    preds = pipe(text)
    ents = []
    for p in preds:
        if p['entity_group'].upper() in ["CHEMICAL", "DISEASE"]:
            ents.append({
                "start": p['start'], "end": p['end'], "label": p['entity_group'].upper(),
                "text": text[p['start']:p['end']]
            })
    return ents


def get_spacy_predictions(text, nlp):
    """Pretrained SpaCy Predictions"""
    doc = nlp(text)
    ents = []
    for ent in doc.ents:
        ents.append({
            "start": ent.start_char, "end": ent.end_char, "label": ent.label_, "text": ent.text
        })
    return ents


def merge_dosage(entities, text):
    """Merges Regex Dosages"""
    existing_ranges = [(e['start'], e['end']) for e in entities]
    dosages = extract_dosages(text)
    for dose in dosages:
        overlap = False
        for start, end in existing_ranges:
            if not (dose['end'] <= start or dose['start'] >= end):
                overlap = True
                break
        if not overlap:
            entities.append(dose)
    return sorted(entities, key=lambda x: x['start'])


def calculate_set_stats(gt_ents, pred_ents):
    """Strict Exact Match Statistics"""
    gt_set = set((e['start'], e['end'], e['label']) for e in gt_ents)
    pred_set = set((e['start'], e['end'], e['label']) for e in pred_ents)

    tp = len(gt_set & pred_set)
    fp = len(pred_set - gt_set)
    fn = len(gt_set - pred_set)
    return tp, fp, fn


# ==========================================
# VISUALIZATION
# ==========================================
def generate_report(viz_data, stats, sample_size):
    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <title>Fixed NER Benchmark</title>
        <style>
            body {{ font-family: -apple-system, system-ui, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif; padding: 20px; background: #f8f9fa; }}
            .container {{ max-width: 95%; margin: 0 auto; }}
            h1, h2 {{ text-align: center; color: #333; }}
            .stats-table {{ margin: 0 auto 30px; border-collapse: collapse; width: 70%; background: white; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
            .stats-table th, .stats-table td {{ border: 1px solid #dee2e6; padding: 12px; text-align: center; }}
            .stats-table th {{ background-color: #e9ecef; }}

            .grid-header {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px; font-weight: bold; text-align: center; margin-bottom: 10px; }}
            .row {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px; margin-bottom: 30px; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }}
            .col {{ padding: 10px; border: 1px solid #eee; border-radius: 4px; overflow-wrap: break-word; }}
            .gt-col {{ background-color: #f0fff4; border-color: #c3e6cb; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Clinical NER Benchmark Report (Fixed)</h1>
            <p style="text-align: center;">Statistics calculated on <b>{sample_size}</b> sentences</p>

            <table class="stats-table">
                <tr><th>Metric</th><th>Your Custom Model</th><th>Pretrained Spacy Model</th></tr>
                <tr><td>Precision</td><td>{stats['our_p']:.2%}</td><td>{stats['pre_p']:.2%}</td></tr>
                <tr><td>Recall</td><td>{stats['our_r']:.2%}</td><td>{stats['pre_r']:.2%}</td></tr>
                <tr><td>F1 Score</td><td><b>{stats['our_f1']:.2%}</b></td><td><b>{stats['pre_f1']:.2%}</b></td></tr>
            </table>

            <h2>Random Sample Visualizations ({VISUALIZATION_COUNT})</h2>
            <div class="grid-header">
                <div>Your BioBERT + Regex</div>
                <div>Pretrained Spacy + Regex</div>
                <div style="color: #2f855a;">Ground Truth</div>
            </div>
    """

    for row in viz_data:
        html += f"""
        <div class="row">
            <div class="col">{row['html_ours']}</div>
            <div class="col">{row['html_pre']}</div>
            <div class="col gt-col">{row['html_gt']}</div>
        </div>
        """
    html += "</div></body></html>"
    return html


# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    print("1. Loading Dataset...")
    dataset = load_dataset(DATASET_NAME)

    # --- FIX 1: LOAD DYNAMIC LABELS OR USE CORRECTED FALLBACK ---
    id2label = get_dataset_labels(dataset)
    # ------------------------------------------------------------

    test_data = dataset['test']
    total_samples = int(len(test_data) * DATASET_PERCENTAGE)

    # Shuffle and select
    indices = list(range(len(test_data)))
    random.shuffle(indices)
    subset = test_data.select(indices[:total_samples])

    print("2. Loading Models...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(YOUR_MODEL_PATH)
        model = AutoModelForTokenClassification.from_pretrained(YOUR_MODEL_PATH)
        pipe = pipeline("token-classification", model=model, tokenizer=tokenizer, aggregation_strategy="simple",
                        device=-1)

        if not spacy.util.is_package(PRETRAINED_MODEL):
            from spacy.cli import download

            download(PRETRAINED_MODEL)
        nlp = spacy.load(PRETRAINED_MODEL)
    except Exception as e:
        print(f"Error: {e}")
        exit()

    print(f"3. Running Inference on {total_samples} sentences...")

    stats_ours = {"tp": 0, "fp": 0, "fn": 0}
    stats_pre = {"tp": 0, "fp": 0, "fn": 0}
    viz_data = []

    for idx, row in tqdm(enumerate(subset), total=total_samples):
        # A. Ground Truth (Using Corrected Logic)
        text, gt_ents_raw = reconstruct_and_get_gt(row['tokens'], row['tags'], id2label)
        gt_ents = merge_dosage(gt_ents_raw, text)

        # B. Predictions
        our_ents = merge_dosage(get_model_predictions(text, pipe), text)
        pre_ents = merge_dosage(get_spacy_predictions(text, nlp), text)

        # C. Update Stats
        tp, fp, fn = calculate_set_stats(gt_ents, our_ents)
        stats_ours['tp'] += tp;
        stats_ours['fp'] += fp;
        stats_ours['fn'] += fn

        tp, fp, fn = calculate_set_stats(gt_ents, pre_ents)
        stats_pre['tp'] += tp;
        stats_pre['fp'] += fp;
        stats_pre['fn'] += fn

        # D. Save for Visualization
        if idx < VISUALIZATION_COUNT:
            opts = {"colors": COLORS}
            viz_data.append({
                "html_ours": displacy.render({"text": text, "ents": our_ents}, style="ent", manual=True, options=opts,
                                             page=False),
                "html_pre": displacy.render({"text": text, "ents": pre_ents}, style="ent", manual=True, options=opts,
                                            page=False),
                "html_gt": displacy.render({"text": text, "ents": gt_ents}, style="ent", manual=True, options=opts,
                                           page=False)
            })


    def calc_metrics(s):
        p = s['tp'] / (s['tp'] + s['fp']) if (s['tp'] + s['fp']) > 0 else 0
        r = s['tp'] / (s['tp'] + s['fn']) if (s['tp'] + s['fn']) > 0 else 0
        f1 = 2 * (p * r) / (p + r) if (p + r) > 0 else 0
        return p, r, f1


    op, or_, of1 = calc_metrics(stats_ours)
    pp, pr, pf1 = calc_metrics(stats_pre)

    final_stats = {
        "our_p": op, "our_r": or_, "our_f1": of1,
        "pre_p": pp, "pre_r": pr, "pre_f1": pf1
    }

    print("4. Generating Report...")
    html_content = generate_report(viz_data, final_stats, total_samples)
    Path("ner_benchmark_fixed.html").write_text(html_content, encoding="utf-8")

    print("\n" + "=" * 60)
    print(f"Results on {total_samples} sentences:")
    print(f"Your Model F1: {of1:.2%}")
    print(f"Spacy Model F1: {pf1:.2%}")
    print(f"Check the corrected report: {Path('ner_benchmark_fixed.html').absolute()}")
    print("=" * 60)