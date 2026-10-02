"""
Publish trained models to one Hugging Face model repo, one folder per model.
"""

import argparse
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi

REPO_NAME = "6emotions"
MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
# Inference files only; DistilBERT's training_args.bin is a pickle and is left out
MODEL_FILES = {
    "logreg": ["6emotions_model.skops"],
    "mlp": ["6emotions_model.skops"],
    "distilbert": ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"],
}


def _operations(name: str) -> list[CommitOperationAdd]:
    model_dir = MODELS_DIR / name
    files = {f"{name}/README.md": model_dir / "README.md"}
    files.update({f"{name}/{file}": model_dir / "artifacts" / file for file in MODEL_FILES[name]})
    files.update({f"{name}/results/{path.name}": path
                  for path in sorted((model_dir / "results").iterdir()) if path.is_file()})
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing files for {name}: {', '.join(missing)}")
    return [CommitOperationAdd(path_in_repo=repo_path, path_or_fileobj=path)
            for repo_path, path in files.items()]


def upload_models(names: list[str], repo_id: str | None = None):
    api = HfApi()
    # Default to the logged-in account, so no username is hardcoded
    repo_id = repo_id or f"{api.whoami()['name']}/{REPO_NAME}"
    operations = [CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=MODELS_DIR / "README.md")]
    for name in names:
        operations += _operations(name)

    api.create_repo(repo_id=repo_id, repo_type="model", private=False, exist_ok=True)
    api.create_commit(
        repo_id=repo_id,
        operations=operations,
        commit_message=f"Upload {', '.join(names)}",
    )
    print(f"Uploaded {', '.join(names)} to https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="*", choices=list(MODEL_FILES), default=list(MODEL_FILES),
                        help="models to upload (default: all)")
    parser.add_argument("--repo-id", help=f"target repo (default: <logged-in user>/{REPO_NAME})")
    args = parser.parse_args()
    upload_models(args.models, args.repo_id)
