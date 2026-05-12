# gizzbrain/cli.py

import argparse
import json
import os
import pandas as pd
from gizzbrain import tagger, encoder, model


def _build_splits(df_original):
    """Return (train_mp3s, val_mp3s, label_map, top_songs) using vanguard top-10 + file-level 80/20 split."""
    top_songs = df_original['title'].value_counts().head(10).index.tolist()
    df_vanguard = df_original[df_original['title'].isin(top_songs)].copy()
    label_map = {title: idx for idx, title in enumerate(top_songs)}

    train_dfs, val_dfs = [], []
    for song in top_songs:
        song_df = df_vanguard[df_vanguard['title'] == song]
        song_df = song_df.sample(frac=1, random_state=42).reset_index(drop=True)
        total_files = len(song_df)
        if total_files > 1:
            split_idx = int(total_files * 0.8)
            if split_idx == total_files:
                split_idx = total_files - 1
            elif split_idx == 0:
                split_idx = 1
        else:
            split_idx = 1
        train_dfs.append(song_df.iloc[:split_idx])
        val_dfs.append(song_df.iloc[split_idx:])

    train_mp3s = pd.concat(train_dfs)
    val_mp3s = pd.concat(val_dfs) if any(len(d) > 0 for d in val_dfs) else pd.DataFrame()
    return train_mp3s, val_mp3s, label_map, top_songs


def _print_eval_report(result_df, inv_label_map, split_name):
    result_df = result_df.copy()
    result_df['predicted_title'] = result_df['predicted_label'].map(inv_label_map)

    chunk_correct = (result_df['label'] == result_df['predicted_label']).sum()
    chunk_total = len(result_df)
    chunk_acc = 100 * chunk_correct / chunk_total if chunk_total else 0

    file_rows = []
    for path, grp in result_df.groupby('path'):
        true_title = grp['title'].iloc[0]
        pred_title = grp['predicted_title'].value_counts().idxmax()
        file_rows.append({
            'true_title': true_title,
            'predicted_title': pred_title,
            'correct': true_title == pred_title,
            'avg_confidence': grp['confidence'].mean() * 100,
        })
    file_df = pd.DataFrame(file_rows)
    file_correct = file_df['correct'].sum()
    file_total = len(file_df)
    file_acc = 100 * file_correct / file_total if file_total else 0

    sep = "=" * 62
    print(f"\n{sep}")
    print(f" GizzBrain Evaluation Report")
    print(f" Split: {split_name}  |  Files: {file_total}  |  Chunks: {chunk_total}")
    print(sep)
    print(f"\n Chunk Accuracy :  {chunk_acc:5.1f}%   ({chunk_correct} / {chunk_total} correct)")
    print(f" File Accuracy  :  {file_acc:5.1f}%   ({file_correct} / {file_total} correctly tagged)\n")

    print(" Per-Song Breakdown:")
    print(f"   {'Song':<33} {'Files':>5}  {'Correct':>7}  {'Acc':>5}  {'Avg Conf':>9}")
    print("   " + "-" * 62)
    for song in sorted(file_df['true_title'].unique()):
        rows = file_df[file_df['true_title'] == song]
        n = len(rows)
        c = rows['correct'].sum()
        acc = 100 * c / n if n else 0
        conf = rows['avg_confidence'].mean()
        print(f"   {song:<33} {n:>5}  {c:>7}  {acc:>4.0f}%  {conf:>8.1f}%")

    wrong = file_df[~file_df['correct']]
    if len(wrong):
        print("\n Top Confusions (file level):")
        confusions = wrong.groupby(['true_title', 'predicted_title']).size().reset_index(name='count')
        confusions = confusions.sort_values('count', ascending=False)
        for _, row in confusions.iterrows():
            n = row['count']
            print(f"   \"{row['true_title']}\"  →  \"{row['predicted_title']}\"  ({n} file{'s' if n > 1 else ''})")
    else:
        print("\n No confusions — all files correctly tagged!")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="GizzBrain: Neural Net MP3 Annotation Tool for Live King Gizzard and The Lizard Wizard"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- COMMAND: SCAN ---
    scan_parser = subparsers.add_parser("scan", help="Scan a directory and create a metadata library")
    scan_parser.add_argument("directory", type=str, help="Path to the directory containing MP3 files")
    scan_parser.add_argument("--output", type=str, default="library.parquet", help="Output filename (default: library.parquet)")

    # --- COMMAND: TRAIN ---
    train_parser = subparsers.add_parser("train", help="Train the neural network on the parsed library")
    train_parser.add_argument("--data", type=str, required=True, help="Path to the library.parquet file")
    train_parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    train_parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")

    # --- COMMAND: EVALUATE ---
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a trained model and print accuracy statistics")
    eval_parser.add_argument("--data", type=str, required=True, help="Path to library.parquet")
    eval_parser.add_argument("--weights", type=str, default="gizzbrain_weights.pt", help="Path to model weights (default: gizzbrain_weights.pt)")
    eval_parser.add_argument("--labels", type=str, default="label_map.json", help="Path to label map JSON (default: label_map.json)")
    eval_parser.add_argument("--split", type=str, default="val", choices=["train", "val", "all"],
                             help="Which data split to evaluate: val (default), train, or all")
    eval_parser.add_argument("--batch-size", type=int, default=32)

    args = parser.parse_args()

    # --- SCAN ---
    if args.command == "scan":
        print(f"Scanning directory: {args.directory}...")
        mp3_files = tagger.find_files(args.directory)
        if not mp3_files:
            print("No MP3 files found in the specified directory.")
            return
        print(f"Found {len(mp3_files)} files. Parsing metadata...")
        df = tagger.convert_metadata(mp3_files)
        df_plain = pd.DataFrame(df.to_dict(orient='list'))
        df_plain.to_parquet(args.output, engine='pyarrow')
        print(f"Library saved successfully to {args.output}")

    # --- TRAIN ---
    elif args.command == "train":
        import torch
        print(f"Loading metadata from {args.data}...")
        df_original = pd.read_parquet(args.data)

        train_mp3s, val_mp3s, label_map, top_songs = _build_splits(df_original)
        print(f"Vanguard Test: Targeting Top 10 songs: {top_songs}")
        print(f"Split completed. Training MP3s: {len(train_mp3s)} | Validation MP3s: {len(val_mp3s)}")

        print("Chunking training audio (this may take a few minutes)...")
        train_chunks = encoder.create_chunk_index(train_mp3s, chunk_length=5.0)
        print("Chunking validation audio...")
        val_chunks = encoder.create_chunk_index(val_mp3s, chunk_length=5.0)

        train_dataset = encoder.GizzDataset(train_chunks, label_mapping=label_map)
        val_dataset = encoder.GizzDataset(val_chunks, label_mapping=label_map)

        trained_model = model.train_model(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )

        torch.save(trained_model.state_dict(), "gizzbrain_weights.pt")
        with open("label_map.json", "w") as f:
            json.dump(label_map, f, indent=2)
        print("Training complete. Weights saved to gizzbrain_weights.pt | Label map saved to label_map.json")

    # --- EVALUATE ---
    elif args.command == "evaluate":
        import torch
        print(f"Loading metadata from {args.data}...")
        df_original = pd.read_parquet(args.data)

        train_mp3s, val_mp3s, label_map, _ = _build_splits(df_original)

        if os.path.exists(args.labels):
            with open(args.labels) as f:
                label_map = json.load(f)
        else:
            print(f"Warning: {args.labels} not found — reconstructing label map from data. Re-train to save a permanent copy.")

        inv_label_map = {v: k for k, v in label_map.items()}
        num_classes = len(label_map)

        if args.split == "train":
            eval_mp3s = train_mp3s
        elif args.split == "val":
            eval_mp3s = val_mp3s
        else:
            eval_mp3s = pd.concat([train_mp3s, val_mp3s])

        if eval_mp3s.empty:
            print("No files in the selected split.")
            return

        print(f"Chunking {args.split} audio...")
        eval_chunks = encoder.create_chunk_index(eval_mp3s, chunk_length=5.0)
        eval_dataset = encoder.GizzDataset(eval_chunks, label_mapping=label_map)

        device = model.get_hardware_device()
        net = model.AudioClassifier(num_classes=num_classes)
        net.load_state_dict(torch.load(args.weights, map_location=device))
        net.to(device)
        print(f"Loaded weights from {args.weights}\n")

        result_df = model.run_inference(eval_dataset, net, device, batch_size=args.batch_size)
        _print_eval_report(result_df, inv_label_map, args.split)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
