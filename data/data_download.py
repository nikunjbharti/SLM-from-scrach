from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Modotte/CodeX-7M-Non-Thinking",
    repo_type="dataset",
    local_dir="./CodeX-7M-Non-Thinking",
)