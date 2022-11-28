import torch

from hw_tts.contrib.text import text_to_sequence


TEXT_PROMPTS = [
    "A defibrillator is a device that gives a high energy electric shock to the heart of someone who is in cardiac arrest",
    "Massachusetts Institute of Technology may be best known for its math, science and engineering education",
    "Wasserstein distance or Kantorovich Rubinstein metric is a distance function defined between probability distributions on a given metric space",
]
ALPHA_GRID = [0.8, 1.0, 1.2]
TEXT_CLEANERS = ["english_cleaners"]


def get_batch(alpha, pitch_alpha, energy_alpha, raw_text, text_id):
    text = torch.tensor(text_to_sequence(raw_text, TEXT_CLEANERS)).long()

    return {
        "raw_text": [raw_text],
        "text_id": text_id,
        "text": text.unsqueeze(0),
        "src_pos": torch.arange(1, text.shape[0] + 1, dtype=torch.long).unsqueeze(0),
        "alpha": alpha,
        "pitch_alpha": pitch_alpha,
        "energy_alpha": energy_alpha,
    }

def get_test_data():
    batches = []
    for alpha in ALPHA_GRID:
        for pitch_alpha in ALPHA_GRID:
            for energy_alpha in ALPHA_GRID:
                for text_id, text in enumerate(TEXT_PROMPTS):
                    batches.append(get_batch(alpha, pitch_alpha, energy_alpha, text, text_id))
    return batches
