from pathlib import Path

from huggingface_hub import HfApi

def upload_model(model_path: str, repo_id: str):
    api = HfApi()
    api.create_repo(repo_id=repo_id, exist_ok=True)
    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo=model_path.split("/")[-1],
        repo_id=repo_id,
    )
    print(f"Model uploaded to https://huggingface.co/{repo_id}")

if __name__ == "__main__":
    upload_model(
        model_path=str(Path(__file__).resolve().parents[1] / "models/logreg/artifacts/6emotions_model.skops"),
        repo_id="<username>/6emotions-classifier"
    )
