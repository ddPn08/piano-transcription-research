from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="ddPn08/maestro-v3.0.0",
    repo_type="dataset",
    local_dir="/mnt/nfs/WD20EARZ-0001/data/ai/datasets/maestro-v3.0.0",
    local_dir_use_symlinks=False,
)
