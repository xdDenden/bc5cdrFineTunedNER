import re
import io
import csv
import base64
import datetime
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from spacy import displacy
from pathlib import Path
from datasets import load_dataset
from torchcrf import CRF  # pip install pytorch-crf
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    RobertaPreTrainedModel,
    RobertaModel,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    pipeline
)
from tqdm import tqdm
import evaluate
from RobertaWithCRF import RobertaWithCRF

# ==========================================
# CONFIGURATION
# ==========================================
DATASET_NAME = "tner/bc5cdr"
BASE_MODEL_FOR_TRAINING = "roberta-base"
YOUR_MODEL_PATH = "./final_clinical_ner_crf_model"
PRETRAINED_MODEL = "tner/roberta-large-bc5cdr"

DATASET_PERCENTAGE = 0.5
RANDOM_DATASET_VIZ_COUNT = 5

CUSTOM_SENTENCES = [
    "The patient was prescribed 50mg of Aspirin for the headache.",
    "Significant side effects were noted after administering 10ml of Doxorubicin.",
    "History of myocardial infarction and hypertension.",
    "Injection of 0.5ml epinephrine resolved the anaphylaxis immediately."
]

COLORS = {
    "CHEMICAL": "linear-gradient(90deg, #aa9cfc, #fc9ce7)",
    "DISEASE": "linear-gradient(90deg, #ff9a8d, #ff6961)",
    "DOSAGE": "linear-gradient(90deg, #feca57, #ff9ff3)",
}


# ==========================================
# PART 2: DATA PREPARATION
# ==========================================

def get_dataset_labels_forced():
    # Force T-NER Schema
    return {0: 'O', 1: 'B-CHEMICAL', 2: 'B-DISEASE', 3: 'I-DISEASE', 4: 'I-CHEMICAL'}


def align_labels_with_tokens(labels, word_ids):
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            current_word = word_id
            label = -100 if word_id is None else labels[word_id]
            new_labels.append(label)
        elif word_id is None:
            new_labels.append(-100)
        else:
            label = labels[word_id]
            if label % 2 == 1: label += 1  # Convert B to I if repeated
            new_labels.append(label)
    return new_labels


def tokenize_and_align_labels(examples, tokenizer):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
    tag_key = "ner_tags" if "ner_tags" in examples else "tags"
    all_labels = examples[tag_key]
    new_labels = []
    for i, labels in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        new_labels.append(align_labels_with_tokens(labels, word_ids))
    tokenized_inputs["labels"] = new_labels
    return tokenized_inputs


# ==========================================
# PART 3: TRAINING LOOP
# ==========================================

def train_new_model():
    print(f"\n[TRAIN] Starting training pipeline using {BASE_MODEL_FOR_TRAINING}...")

    # 1. Load Data
    raw_datasets = load_dataset(DATASET_NAME)

    # 2. Setup Labels
    label_list = ["O", "B-CHEMICAL", "B-DISEASE", "I-DISEASE", "I-CHEMICAL"]
    id2label = {i: l for i, l in enumerate(label_list)}
    label2id = {l: i for i, l in enumerate(label_list)}

    # 3. Tokenize
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_FOR_TRAINING, add_prefix_space=True)
    tokenized_datasets = raw_datasets.map(
        lambda x: tokenize_and_align_labels(x, tokenizer),
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
    )

    # 4. Initialize Custom Model (CRF)
    print("[TRAIN] Initializing Custom RobertaWithCRF...")
    model = RobertaWithCRF.from_pretrained(
        BASE_MODEL_FOR_TRAINING,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id,
    )

    # 5. Training Args
    args = TrainingArguments(
        output_dir=YOUR_MODEL_PATH,
        eval_strategy="epoch",
        learning_rate=3e-5,  # Slightly higher for CRF
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        save_strategy="no",
        remove_unused_columns=False  # Important for custom models sometimes
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)
    metric = evaluate.load("seqeval")

    def compute_metrics(p):
        # NOTE: For speed during training, we use argmax on emissions.
        # Ideally, we should use Viterbi here too, but argmax is a good proxy for progress.
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        results = metric.compute(predictions=true_predictions, references=true_labels)
        return {
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("[TRAIN] Training model...")
    trainer.train()

    print(f"[TRAIN] Saving model to {YOUR_MODEL_PATH}...")
    trainer.save_model(YOUR_MODEL_PATH)
    tokenizer.save_pretrained(YOUR_MODEL_PATH)
    print("[TRAIN] Training Complete.")


# ==========================================
# PART 4: INFERENCE & UTILS
# ==========================================

def extract_dosages(text):
    dosage_pattern = r'\b\d+\.?\d*\s?(mg|g|ml|mcg|units?)\b'
    dosages = []
    for match in re.finditer(dosage_pattern, text, re.IGNORECASE):
        dosages.append({"start": match.start(), "end": match.end(), "label": "DOSAGE", "text": match.group()})
    return dosages


def tokens_to_text_and_entities(tokens, tags, id2label):
    text = " ".join(tokens)
    entities = []
    current_entity = None
    char_pos = 0
    for token, tag_id in zip(tokens, tags):
        token_start = char_pos
        token_end = char_pos + len(token)
        label_full = id2label.get(tag_id, "O")

        if label_full.startswith("B-"):
            if current_entity: entities.append(current_entity)
            current_entity = {"start": token_start, "end": token_end, "label": label_full[2:].upper(), "text": token}
        elif label_full.startswith("I-") and current_entity:
            if label_full[2:].upper() == current_entity["label"]:
                current_entity["end"] = token_end
                current_entity["text"] += " " + token
            else:
                entities.append(current_entity)
                current_entity = {"start": token_start, "end": token_end, "label": label_full[2:].upper(),
                                  "text": token}
        else:
            if current_entity: entities.append(current_entity)
            current_entity = None
        char_pos = token_end + 1
    if current_entity: entities.append(current_entity)
    return text, entities


def get_model_predictions_custom(text, model, tokenizer):
    """
    CUSTOM INFERENCE FOR CRF MODEL (Uses Viterbi Decode)
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    # Move to device
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        # returns List[List[int]]
        predicted_tag_ids = model.decode(inputs['input_ids'], inputs['attention_mask'])[0]

    # Convert IDs to Labels
    # We must align subtokens back to words, or just reconstruct roughly for visualization
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

    # Filter out special tokens for visualization mapping
    ents = []
    current_entity = None

    # We reconstruct entities from tokens
    # Note: This is a simplified reconstruction for RoBERTa subtokens
    # In a production system, you would use offset_mapping from the tokenizer

    reconstructed_text = ""
    char_map = []  # maps token index to (start, end) in text

    # Simple reconstruction for display (Pipeline handles this better usually, but we are manual now)
    # We will trust the input text and simple offset matching for this demo

    token_offsets = []
    cursor = 0

    # Create offsets for raw text mapping
    # This is tricky with RoBERTa's "Ġ". We will assume simple mapping for this demo script.
    # Better approach: Use the tokenizer's offset_mapping

    inputs_with_offsets = tokenizer(text, return_offsets_mapping=True, return_tensors="pt")
    offsets = inputs_with_offsets["offset_mapping"][0].tolist()

    for idx, (tag_id, (start, end)) in enumerate(zip(predicted_tag_ids, offsets)):
        if start == end: continue  # Special token

        label_full = model.config.id2label[tag_id]
        if label_full == "O":
            if current_entity: ents.append(current_entity); current_entity = None
            continue

        label_type = label_full[2:]  # Remove B- or I-

        if label_full.startswith("B-"):
            if current_entity: ents.append(current_entity)
            current_entity = {"start": start, "end": end, "label": label_type, "text": text[start:end]}

        elif label_full.startswith("I-"):
            if current_entity and current_entity['label'] == label_type:
                current_entity['end'] = end
                current_entity['text'] = text[current_entity['start']:end]
            else:
                # If I- tag appears without B-, treat as new B- (CRF usually prevents this, but good safety)
                if current_entity: ents.append(current_entity)
                current_entity = {"start": start, "end": end, "label": label_type, "text": text[start:end]}

    if current_entity: ents.append(current_entity)
    return ents


def get_model_predictions_pipeline(text, pipe):
    """
    Standard Pipeline Inference (For the Pretrained Baseline)
    """
    preds = pipe(text)
    ents = []
    for p in preds:
        label = p['entity_group'].upper()
        if label in ["CHEMICAL", "DISEASE"]:
            ents.append({"start": p['start'], "end": p['end'], "label": label, "text": text[p['start']:p['end']]})
    return ents


def merge_dosage_for_viz(entities, text):
    viz_entities = [e.copy() for e in entities]
    existing_ranges = [(e['start'], e['end']) for e in viz_entities]
    dosages = extract_dosages(text)
    for dose in dosages:
        overlap = False
        for start, end in existing_ranges:
            if not (dose['end'] <= start or dose['start'] >= end):
                overlap = True;
                break
        if not overlap: viz_entities.append(dose)
    return sorted(viz_entities, key=lambda x: x['start'])


def has_overlap(ent1, ent2):
    if ent1['label'] != ent2['label']: return False
    return max(ent1['start'], ent2['start']) < min(ent1['end'], ent2['end'])


def calculate_smart_stats(gt_ents, pred_ents, stats_accumulator):
    matched_gt, matched_pred = set(), set()
    for gt_idx, gt in enumerate(gt_ents):
        for pred_idx, pred in enumerate(pred_ents):
            if pred_idx in matched_pred: continue
            if has_overlap(gt, pred):
                stats_accumulator[gt['label']]['tp'] += 1
                matched_gt.add(gt_idx)
                matched_pred.add(pred_idx)
                break
    for gt_idx, gt in enumerate(gt_ents):
        if gt_idx not in matched_gt: stats_accumulator[gt['label']]['fn'] += 1
    for pred_idx, pred in enumerate(pred_ents):
        if pred_idx not in matched_pred: stats_accumulator[pred['label']]['fp'] += 1
    return stats_accumulator


# ==========================================
# PART 5: REPORTING
# ==========================================

def save_csv_report(folder_path, stats_ours, stats_pre):
    file_path = folder_path / "statistics.csv"

    def get_metrics(s):
        tp, fp, fn = s['tp'], s['fp'], s['fn']
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (p * r) / (p + r) if (p + r) > 0 else 0
        return tp, fp, fn, p, r, f1

    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Model", "Label", "TP", "FP", "FN", "Precision", "Recall", "F1 Score"])
        for label in ["CHEMICAL", "DISEASE"]:
            tp, fp, fn, p, r, f1 = get_metrics(stats_ours[label])
            writer.writerow(["Your Custom CRF Model", label, tp, fp, fn, f"{p:.4f}", f"{r:.4f}", f"{f1:.4f}"])
            tp, fp, fn, p, r, f1 = get_metrics(stats_pre[label])
            writer.writerow(["Pretrained RoBERTa", label, tp, fp, fn, f"{p:.4f}", f"{r:.4f}", f"{f1:.4f}"])
    print(f"Stats saved to: {file_path}")


def create_plots(stats_ours, stats_pre):
    labels = ["CHEMICAL", "DISEASE"]

    def get_f1(s):
        p = s['tp'] / (s['tp'] + s['fp']) if (s['tp'] + s['fp']) > 0 else 0
        r = s['tp'] / (s['tp'] + s['fn']) if (s['tp'] + s['fn']) > 0 else 0
        return 2 * (p * r) / (p + r) if (p + r) > 0 else 0

    f1_ours = [get_f1(stats_ours[l]) for l in labels]
    f1_pre = [get_f1(stats_pre[l]) for l in labels]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(labels));
    width = 0.35
    ax.bar(x - width / 2, f1_ours, width, label='Your CRF Model', color='#6c5ce7')
    ax.bar(x + width / 2, f1_pre, width, label='Pretrained RoBERTa', color='#0984e3')
    ax.set_ylabel('F1 Score');
    ax.set_title('Model F1 Score Comparison')
    ax.set_xticks(x);
    ax.set_xticklabels(labels);
    ax.legend();
    ax.grid(axis='y', alpha=0.3)
    img_buf = io.BytesIO();
    plt.savefig(img_buf, format='png', bbox_inches='tight')
    img_buf.seek(0);
    plot1_b64 = base64.b64encode(img_buf.read()).decode('utf-8');
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 5))
    fps = [sum(stats_ours[l]['fp'] for l in labels), sum(stats_pre[l]['fp'] for l in labels)]
    fns = [sum(stats_ours[l]['fn'] for l in labels), sum(stats_pre[l]['fn'] for l in labels)]
    models = ['Your CRF Model', 'Pretrained RoBERTa']
    ax.bar(models, fns, label='Missed Entity (FN)', color='#ff7675')
    ax.bar(models, fps, bottom=fns, label='Hallucination (FP)', color='#ffeaa7')
    ax.set_ylabel('Total Errors');
    ax.set_title('Error Distribution');
    ax.legend()
    img_buf = io.BytesIO();
    plt.savefig(img_buf, format='png', bbox_inches='tight')
    img_buf.seek(0);
    plot2_b64 = base64.b64encode(img_buf.read()).decode('utf-8');
    plt.close()

    return plot1_b64, plot2_b64


def generate_report(custom_data, dataset_data, stats_ours, stats_pre, sample_size, plot1, plot2):
    def calc_macro(stats_dict):
        tp = sum(stats_dict[l]['tp'] for l in ["CHEMICAL", "DISEASE"])
        fp = sum(stats_dict[l]['fp'] for l in ["CHEMICAL", "DISEASE"])
        fn = sum(stats_dict[l]['fn'] for l in ["CHEMICAL", "DISEASE"])
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (p * r) / (p + r) if (p + r) > 0 else 0
        return p, r, f1

    op, or_, of1 = calc_macro(stats_ours)
    pp, pr, pf1 = calc_macro(stats_pre)

    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Clinical NER Report</title>
        <style>
            body {{ font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background-color: #f4f6f9; margin: 0; padding: 0; color: #333; }}
            .header {{ background: linear-gradient(135deg, #6c5ce7, #a29bfe); color: white; padding: 40px 20px; text-align: center; margin-bottom: 40px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
            .header h1 {{ margin: 0; font-size: 2.5em; }}
            .header p {{ margin-top: 10px; font-size: 1.2em; opacity: 0.9; }}
            .container {{ max-width: 1100px; margin: 0 auto; padding: 0 20px; }}
            .stats-card {{ background: white; padding: 25px; border-radius: 12px; box-shadow: 0 2px 10px rgba(0,0,0,0.05); margin-bottom: 40px; }}
            table {{ width: 100%; border-collapse: collapse; text-align: center; }}
            th {{ background-color: #f8f9fa; color: #555; font-weight: 600; padding: 12px; border-bottom: 2px solid #eee; }}
            td {{ padding: 12px; border-bottom: 1px solid #eee; font-size: 1.05em; }}
            .highlight {{ color: #2d3436; font-weight: bold; }}
            .charts-row {{ display: flex; gap: 20px; margin-bottom: 40px; flex-wrap: wrap; }}
            .chart-box {{ flex: 1; background: white; padding: 20px; border-radius: 12px; box-shadow: 0 2px 10px rgba(0,0,0,0.05); text-align: center; min-width: 300px; }}
            .chart-box img {{ max-width: 100%; height: auto; }}
            .section-title {{ font-size: 1.5em; margin-bottom: 20px; color: #2d3436; border-left: 5px solid #6c5ce7; padding-left: 15px; }}
            .viz-card {{ background: white; border-radius: 12px; box-shadow: 0 2px 10px rgba(0,0,0,0.05); margin-bottom: 25px; overflow: hidden; }}
            .viz-header {{ display: flex; background: #f8f9fa; border-bottom: 1px solid #eee; }}
            .viz-col-header {{ flex: 1; text-align: center; padding: 10px; font-weight: 600; color: #636e72; font-size: 0.9em; text-transform: uppercase; }}
            .viz-content {{ display: flex; }}
            .viz-col {{ flex: 1; padding: 20px; border-right: 1px solid #f1f1f1; font-size: 15px; line-height: 1.6; }}
            .viz-col:last-child {{ border-right: none; }}
            .legend {{ text-align: center; margin-bottom: 30px; font-size: 0.9em; }}
            .dot {{ height: 10px; width: 10px; border-radius: 50%; display: inline-block; margin-right: 5px; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Clinical NER Benchmark Report (With CRF)</h1>
            <p>Comparing Custom RoBERTa-CRF vs. Pretrained RoBERTa-Large on {sample_size} BC5CDR samples</p>
        </div>
        <div class="container">
            <div class="stats-card">
                <table>
                    <tr><th>Metric (Macro Avg)</th><th>Your Custom CRF Model</th><th>Pretrained Benchmark (Large)</th></tr>
                    <tr><td>Precision</td><td>{op:.2%}</td><td>{pp:.2%}</td></tr>
                    <tr><td>Recall</td><td>{or_:.2%}</td><td>{pr:.2%}</td></tr>
                    <tr class="highlight"><td>F1 Score</td><td>{of1:.2%}</td><td>{pf1:.2%}</td></tr>
                </table>
            </div>
            <div class="charts-row">
                <div class="chart-box"><h3>Performance</h3><img src="data:image/png;base64,{plot1}" /></div>
                <div class="chart-box"><h3>Error Analysis</h3><img src="data:image/png;base64,{plot2}" /></div>
            </div>
            <div class="legend">
                <span style="margin-right:15px"><span class="dot" style="background:#aa9cfc"></span>Chemical</span>
                <span style="margin-right:15px"><span class="dot" style="background:#ff9a8d"></span>Disease</span>
                <span><span class="dot" style="background:#feca57"></span>Dosage (Regex)</span>
            </div>
            <div class="section-title">Custom Test Sentences</div>
            """
    for row in custom_data:
        html += f"""<div class="viz-card"><div class="viz-header"><div class="viz-col-header" style="color:#6c5ce7">Your Model</div><div class="viz-col-header" style="color:#0984e3">Pretrained RoBERTa</div></div><div class="viz-content"><div class="viz-col">{row['html_ours']}</div><div class="viz-col">{row['html_pre']}</div></div></div>"""

    html += """<br><div class="section-title">Random Dataset Samples</div>"""
    for row in dataset_data:
        html += f"""<div class="viz-card"><div class="viz-header"><div class="viz-col-header" style="color:#6c5ce7">Your Model</div><div class="viz-col-header" style="color:#0984e3">Pretrained RoBERTa</div></div><div class="viz-content"><div class="viz-col">{row['html_ours']}</div><div class="viz-col">{row['html_pre']}</div></div></div>"""

    html += f"""<div style="text-align:center; padding: 20px; color: #aaa; font-size: 0.8em;">Generated by Gemini • {datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}</div></div></body></html>"""
    return html


# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    report_dir = Path(f"./reports/{timestamp}")
    report_dir.mkdir(parents=True, exist_ok=True)
    print(f"Directory created: {report_dir}")

    while True:
        choice = input("Do you want to (T)rain a new model or (L)oad the existing one? [T/L]: ").strip().upper()
        if choice in ['T', 'L']: break

    if choice == 'T':
        train_new_model()

    print("\n2. Loading Models...")
    try:
        # Load CUSTOM model using the Custom Class
        tokenizer_ours = AutoTokenizer.from_pretrained(YOUR_MODEL_PATH)
        model_ours = RobertaWithCRF.from_pretrained(YOUR_MODEL_PATH)
        # Note: We do NOT use the Pipeline for the custom model because Pipeline doesn't support the custom CRF decode method easily.
        # We will use get_model_predictions_custom instead.

        # Load PRETRAINED model using Standard Pipeline
        tokenizer_pre = AutoTokenizer.from_pretrained(PRETRAINED_MODEL)
        model_pre = AutoModelForTokenClassification.from_pretrained(PRETRAINED_MODEL)
        pipe_pre = pipeline("token-classification", model=model_pre, tokenizer=tokenizer_pre,
                            aggregation_strategy="simple")
    except Exception as e:
        print(f"Error loading models: {e}");
        exit()

    print(f"3. Processing Custom Sentences...")
    custom_viz_data = []
    opts = {"colors": COLORS}

    for text in CUSTOM_SENTENCES:
        # Use CUSTOM function for our model
        our_preds = get_model_predictions_custom(text, model_ours, tokenizer_ours)
        # Use PIPELINE for pre model
        pre_preds = get_model_predictions_pipeline(text, pipe_pre)

        custom_viz_data.append({
            "html_ours": displacy.render({"text": text, "ents": merge_dosage_for_viz(our_preds, text)}, style="ent",
                                         manual=True, options=opts, page=False),
            "html_pre": displacy.render({"text": text, "ents": merge_dosage_for_viz(pre_preds, text)}, style="ent",
                                        manual=True, options=opts, page=False)
        })

    print("4. Benchmarking Dataset...")
    dataset = load_dataset(DATASET_NAME)
    id2label = get_dataset_labels_forced()
    tag_key = "ner_tags" if "ner_tags" in dataset['test'].features else "tags"
    subset = dataset['test'].shuffle().select(range(int(len(dataset['test']) * DATASET_PERCENTAGE)))

    labels = ["CHEMICAL", "DISEASE"]
    stats_ours = {l: {'tp': 0, 'fp': 0, 'fn': 0} for l in labels}
    stats_pre = {l: {'tp': 0, 'fp': 0, 'fn': 0} for l in labels}
    dataset_viz_data = []

    for idx, row in tqdm(enumerate(subset), total=len(subset)):
        text, gt_ents = tokens_to_text_and_entities(row['tokens'], row[tag_key], id2label)

        # Inference Custom
        our_preds = get_model_predictions_custom(text, model_ours, tokenizer_ours)
        # Inference Pretrained
        pre_preds = get_model_predictions_pipeline(text, pipe_pre)

        stats_ours = calculate_smart_stats(gt_ents, our_preds, stats_ours)
        stats_pre = calculate_smart_stats(gt_ents, pre_preds, stats_pre)

        if idx < RANDOM_DATASET_VIZ_COUNT:
            dataset_viz_data.append({
                "html_ours": displacy.render({"text": text, "ents": merge_dosage_for_viz(our_preds, text)}, style="ent",
                                             manual=True, options=opts, page=False),
                "html_pre": displacy.render({"text": text, "ents": merge_dosage_for_viz(pre_preds, text)}, style="ent",
                                            manual=True, options=opts, page=False)
            })

    print("\n5. Generating Reports...")
    plot1, plot2 = create_plots(stats_ours, stats_pre)
    html_content = generate_report(custom_viz_data, dataset_viz_data, stats_ours, stats_pre, len(subset), plot1, plot2)

    (report_dir / "report.html").write_text(html_content, encoding="utf-8")
    save_csv_report(report_dir, stats_ours, stats_pre)

    print(f"\n[DONE] Report and CSV saved to: {report_dir.absolute()}")