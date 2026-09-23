"""Manual smoke test: run the trained model on three hand-written abstracts.

The examples are written for this script and are not from the dataset, so a
sensible answer shows the model generalises rather than recalls.

Run: python scripts/test_inference_manual.py
"""

from arxiv_classifier.inference import Predictor

MODEL_DIR = "data/processed/transformer_full_checkpoints/final"
EDA_STATS_PATH = "data/processed/eda_stats.json"

EXAMPLES = [
    (
        "Deep Reinforcement Learning for Robotic Manipulation",
        (
            "We propose a novel deep reinforcement learning approach for robotic "
            "manipulation tasks. Our method combines a convolutional neural network "
            "policy with a physics-based simulator to learn dexterous grasping "
            "behaviors from raw pixel observations, achieving state-of-the-art "
            "sample efficiency on a suite of benchmark tasks."
        ),
    ),
    (
        "A New Method for Solving the Navier-Stokes Equations",
        (
            "This paper presents a finite element method for numerically solving "
            "the incompressible Navier-Stokes equations. We derive stability "
            "bounds for the proposed discretization scheme and demonstrate "
            "convergence on standard computational fluid dynamics benchmarks."
        ),
    ),
    (
        "Observational Constraints on Dark Matter Halo Profiles",
        (
            "We use gravitational lensing observations from a large galaxy survey "
            "to constrain the density profile of dark matter halos. Our results "
            "are consistent with the NFW profile and provide new bounds on "
            "self-interacting dark matter cross-sections."
        ),
    ),
]

if __name__ == "__main__":
    print(f"Loading model from {MODEL_DIR} ...")
    predictor = Predictor(model_dir=MODEL_DIR, eda_stats_path=EDA_STATS_PATH)
    print(f"Loaded. device={predictor.device}  threshold={predictor.threshold}\n")

    for title, abstract in EXAMPLES:
        text = f"{title} {abstract}"
        predictions = predictor.predict(text)
        print(f"Title: {title}")
        print(f"Predicted categories: {predictions}")
        print()
