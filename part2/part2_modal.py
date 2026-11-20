"""Run the assignment's Part 2 pipeline on a Modal GPU."""

from pathlib import Path

import modal


LOCAL_PART2 = Path(__file__).resolve().parent
REMOTE_PART2 = Path("/opt/assignment-part2")
VOLUME_ROOT = Path("/assignment-volume")
VOLUME_NAME = "dsga1011-a4-part2"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements(LOCAL_PART2 / "requirements.txt")
    .env({
        "HF_HOME": str(VOLUME_ROOT / "hf-cache"),
        "TOKENIZERS_PARALLELISM": "false",
    })
    .add_local_dir(LOCAL_PART2 / "data", str(REMOTE_PART2 / "data"), copy=True)
    .add_local_file(
        LOCAL_PART2 / "records" / "ground_truth_dev.pkl",
        str(REMOTE_PART2 / "ground_truth_dev.pkl"),
        copy=True,
    )
)
for filename in ("load_data.py", "train_t5.py", "t5_utils.py", "utils.py"):
    image = image.add_local_file(
        LOCAL_PART2 / filename, str(REMOTE_PART2 / filename), copy=True
    )

app = modal.App("dsga1011-a4-part2", image=image)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


def restore_checkpoint(archive_path, best_dir):
    """Import only the files in a checkpoint archive's best folder."""
    import shutil
    import tarfile

    if not archive_path.is_file():
        raise FileNotFoundError(f"No trained weights found. Upload {archive_path} or train first.")
    with tarfile.open(archive_path, "r:gz") as archive:
        members = {
            Path(item.name).name: item
            for item in archive.getmembers()
            if item.isfile() and Path(item.name).parent.name == "best"
        }
        required = {"config.json", "model.safetensors"}
        if not required <= members.keys():
            raise ValueError("Checkpoint archive must contain best/config.json and best/model.safetensors.")
        best_dir.mkdir(parents=True, exist_ok=True)
        for name in ("config.json", "generation_config.json", "model.safetensors"):
            if name in members:
                with archive.extractfile(members[name]) as source, (best_dir / name).open("wb") as target:
                    shutil.copyfileobj(source, target)


def compare_decoding(args, model, output):
    import csv
    import json
    import pickle
    from shutil import copyfile
    from zipfile import ZIP_DEFLATED, ZipFile

    import train_t5 as training

    loader = training.get_dataloader(args.test_batch_size, "dev")
    runs = []
    for beams in (1, 4):
        training.DECODE_BEAMS = beams
        sql = Path("results") / f"dev-beam{beams}.sql"
        records = Path("records") / f"dev-beam{beams}.pkl"
        loss, f1, record_em, sql_em, error_rate = training.eval_epoch(
            args, model, loader, "data/dev.sql", sql, "records/ground_truth_dev.pkl", records
        )
        metrics = dict(beams=beams, loss=loss, record_f1=float(f1),
                       record_em=record_em, sql_em=sql_em, error_rate=error_rate)
        runs.append((sql, records, metrics))
        print(f"Beams {beams}: dev Record F1 {f1:.2%}, Record EM {record_em:.2%}, "
              f"SQL EM {sql_em:.2%}, SQL errors {error_rate:.2%}", flush=True)

    sql, records, best = max(runs, key=lambda run: run[2]["record_f1"])
    review_dir = output / "q6-dev"
    review_dir.mkdir(exist_ok=True)
    for src, name in (
        (sql, "predicted_dev.sql"),
        (records, "predicted_dev.pkl"),
        ("data/dev.nl", "dev.nl"),
        ("data/dev.sql", "dev.sql"),
        ("records/ground_truth_dev.pkl", "ground_truth_dev.pkl"),
    ):
        copyfile(src, review_dir / name)
    (review_dir / "metrics.json").write_text(
        json.dumps({"run": args.experiment_name, "selected": best, "comparison": [r[2] for r in runs]},
                   indent=2),
        encoding="utf-8",
    )

    questions = (review_dir / "dev.nl").read_text(encoding="utf-8").splitlines()
    gold_sql = (review_dir / "dev.sql").read_text(encoding="utf-8").splitlines()
    predicted = sql.read_text(encoding="utf-8").splitlines()
    with records.open("rb") as handle:
        predicted_rows, errors = pickle.load(handle)
    with Path("records/ground_truth_dev.pkl").open("rb") as handle:
        gold_rows, _ = pickle.load(handle)
    if len({len(x) for x in (questions, gold_sql, predicted, predicted_rows, errors, gold_rows)}) != 1:
        raise ValueError("Dev queries and records have mismatched counts.")
    with (review_dir / "review.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("example", "question", "reference_sql", "predicted_sql", "sql_error", "records_match"))
        for i, (question, reference, prediction, error, gold, pred) in enumerate(
            zip(questions, gold_sql, predicted, errors, gold_rows, predicted_rows), 1
        ):
            writer.writerow((i, question, reference, prediction, error, set(gold) == set(pred)))
    with ZipFile(output / "q6-dev-files.zip", "w", ZIP_DEFLATED) as archive:
        for path in sorted(review_dir.iterdir()):
            archive.write(path, path.relative_to(output))
    print(f"Selected decoder: {best['beams']} beam(s)", flush=True)
    volume.commit()
    return best["beams"]


def save_submission(args, model, best_dir, output):
    import pickle
    import tarfile
    from zipfile import ZIP_DEFLATED, ZipFile

    import train_t5 as training

    training.generate_submission(args, model)
    sql = Path("results/t5_ft_experiment_test.sql")
    records = Path("records/t5_ft_experiment_test.pkl")
    with records.open("rb") as handle:
        rows, errors = pickle.load(handle)
    expected = len(Path("data/test.nl").read_text(encoding="utf-8").splitlines())
    if not len(sql.read_text(encoding="utf-8").splitlines()) == len(rows) == len(errors) == expected == 432:
        raise ValueError("Expected 432 test inputs, SQL predictions, records, and error entries.")
    with ZipFile(output / "q7-submission.zip", "w", ZIP_DEFLATED) as archive:
        for path in (sql, records):
            archive.write(path, path.name)
    with tarfile.open(output / "q7-checkpoint.tar.gz", "w:gz") as archive:
        archive.add(best_dir, arcname="best")
    volume.commit()
    print("Verified and saved 432 test predictions and records.", flush=True)


@app.function(gpu="A100-40GB", cpu=4, memory=16384, timeout=3 * 60 * 60, volumes={str(VOLUME_ROOT): volume})
def run_part2(
    stage: str = "finish",
    run_name: str = "lr1e3_mps",
    epochs: int = 8,
    batch_size: int = 8,
    inference_batch_size: int = 4,
):
    """Train, compare dev decoders, or generate the final submission."""
    import json
    import os
    import re
    import shutil
    import sys
    from tempfile import TemporaryDirectory
    from types import SimpleNamespace

    import torch

    if stage not in {"train", "dev", "test", "finish"}:
        raise ValueError("Stage must be train, dev, test, or finish (dev followed by test).")
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", run_name):
        raise ValueError("Use 1–64 letters, digits, underscores, or hyphens for run_name.")
    if min(epochs, batch_size, inference_batch_size) < 1:
        raise ValueError("Epochs and batch sizes must be positive.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU unavailable.")
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    output = VOLUME_ROOT / "runs" / run_name
    work_dirs = ("checkpoints", "results", "records")
    for directory in work_dirs:
        (output / directory).mkdir(parents=True, exist_ok=True)
    shutil.copyfile(REMOTE_PART2 / "ground_truth_dev.pkl", output / "records" / "ground_truth_dev.pkl")
    best_dir = output / "checkpoints" / "ft_experiments" / run_name / "best"
    args = SimpleNamespace(
        finetune=True, experiment_name=run_name, learning_rate=1e-3,
        max_n_epochs=epochs, patience_epochs=3, batch_size=batch_size,
        test_batch_size=inference_batch_size, optimizer_type="AdamW",
        scheduler_type="cosine", weight_decay=0, num_warmup_epochs=0, use_wandb=False,
    )
    sys.path.insert(0, str(REMOTE_PART2))
    import train_t5 as training

    previous_dir = Path.cwd()
    with TemporaryDirectory() as temp:
        work = Path(temp)
        (work / "data").symlink_to(REMOTE_PART2 / "data", target_is_directory=True)
        for directory in work_dirs:
            (work / directory).symlink_to(output / directory, target_is_directory=True)
        os.chdir(work)
        try:
            if stage == "train":
                if (best_dir / "model.safetensors").exists():
                    raise FileExistsError("This run has a best checkpoint. Use a new run name to retrain.")
                training.set_random_seeds()
                save_model = training.save_model

                def save_and_commit(directory, model, best):
                    save_model(directory, model, best)
                    if best:
                        volume.commit()

                training.save_model = save_and_commit
                try:
                    train_loader = training.get_dataloader(batch_size, "train")
                    dev_loader = training.get_dataloader(inference_batch_size, "dev")
                    model = training.initialize_model(args)
                    optimizer, scheduler = training.initialize_optimizer_and_scheduler(args, model, len(train_loader))
                    training.train(args, model, train_loader, dev_loader, optimizer, scheduler)
                finally:
                    training.save_model = save_model
                volume.commit()
                print(f"Saved {best_dir}. Run stage finish to generate your files.", flush=True)
                return

            if not (best_dir / "model.safetensors").is_file():
                restore_checkpoint(VOLUME_ROOT / "imports" / f"{run_name}.tar.gz", best_dir)
                volume.commit()
            model = training.load_model_from_checkpoint(args, best=True)
            if stage in {"dev", "finish"}:
                training.DECODE_BEAMS = compare_decoding(args, model, output)
            if stage == "test":
                metrics = output / "q6-dev" / "metrics.json"
                training.DECODE_BEAMS = json.loads(metrics.read_text(encoding="utf-8"))["selected"]["beams"]
            if stage in {"test", "finish"}:
                save_submission(args, model, best_dir, output)
        finally:
            os.chdir(previous_dir)
