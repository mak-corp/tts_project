import torch
from collections import defaultdict

from hw_tts.contrib.text import text_to_sequence


TEXT_PROMPTS = [
    "A defibrillator is a device that gives a high energy electric shock to the heart of someone who is in cardiac arrest",
    "Massachusetts Institute of Technology may be best known for its math, science and engineering education",
    "Wasserstein distance or Kantorovich Rubinstein metric is a distance function defined between probability distributions on a given metric space",
]
TEXT_CLEANERS = ["english_cleaners"]


def get_batch(name, raw_text, text_id, alpha, pitch_alpha=None, energy_alpha=None):
    text = torch.tensor(text_to_sequence(raw_text, TEXT_CLEANERS)).long()

    batch = {
        "name": name,
        "raw_text": raw_text,
        "text_id": text_id,
        "text": text.unsqueeze(0),
        "src_pos": torch.arange(1, text.shape[0] + 1, dtype=torch.long).unsqueeze(0),
        "alpha": alpha,
    }

    if pitch_alpha is not None and energy_alpha is not None:
        batch.update({
            "pitch_alpha": pitch_alpha,
            "energy_alpha": energy_alpha,
        })
    
    return batch


def get_test_data():
    batches = defaultdict(list)
    for text_id, text in enumerate(TEXT_PROMPTS):
        batches[text_id].append(get_batch("usual", text, text_id, alpha=1.0, pitch_alpha=1.0, energy_alpha=1.0))
        batches[text_id].append(get_batch("all=0.8", text, text_id, alpha=0.8, pitch_alpha=0.8, energy_alpha=0.8))
        batches[text_id].append(get_batch("all=1.2", text, text_id, alpha=1.2, pitch_alpha=1.2, energy_alpha=1.2))
        for tp in ["duration", "pitch", "energy"]:
            for value in [0.8, 1.2]:
                alpha, pitch_alpha, energy_alpha = 1.0, 1.0, 1.0
                if tp == "duration":
                    alpha = value
                elif tp == "pitch":
                    pitch_alpha = value
                else:
                    energy_alpha = value
                name = "%s=%.2f" % (tp, value)
                batches[text_id].append(get_batch(name, text, text_id, alpha, pitch_alpha, energy_alpha))
    return batches


def get_v1_test_data():
    batches = defaultdict(list)
    for text_id, text in enumerate(TEXT_PROMPTS):
        for alpha in [0.8, 1.0, 1.2]:
            name = "a=%.2f" % (alpha)
            batches[text_id].append(get_batch(name, text, text_id, alpha))
    return batches
