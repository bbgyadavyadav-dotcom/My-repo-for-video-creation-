
import os
import json
from pathlib import Path
from huggingface_hub import InferenceClient

ROOT = Path(__file__).resolve().parents[1]
CHARACTERS = ROOT / "characters.json"
OUTPUT = ROOT / "references" / "CHAR_001_base.png"

STYLE = (
    "Dark hand-drawn horror manga illustration, rough uneven ink outlines, "
    "dense cross-hatching and stippled shading, painterly textured colour washes, "
    "muted deep forest green, charcoal black and dusky purple palette, "
    "cinematic widescreen composition, eerie low-key lighting, "
    "grounded realistic anatomy, professional seinen horror comic aesthetic."
)

def main():
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN secret is missing.")

    data = json.loads(CHARACTERS.read_text(encoding="utf-8"))
    character = next(
        c for c in data["characters"] if c["id"] == "CHAR_001"
    )

    prompt = (
        STYLE
        + " Create a full-body character reference illustration. "
        "One adult person standing alone in a neutral three-quarter pose, "
        "centered against a simple muted grey-green background. "
        + character["prompt_descriptor"]
        + ". Base gear state only: no mask, no goggles, no plastic suit, "
        "no weapons. No text, labels, watermark or extra people."
    )

    client = InferenceClient(
        model="black-forest-labs/FLUX.1-schnell",
        token=token
    )

    image = client.text_to_image(
        prompt,
        width=768,
        height=432
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    image.save(OUTPUT, format="PNG")
    print("Generated:", OUTPUT)

if __name__ == "__main__":
    main()
  
