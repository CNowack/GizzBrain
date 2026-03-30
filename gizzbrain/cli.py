# gizzbrain/cli.py

import argparse
import os
import pandas as pd
from gizzbrain import tagger, encoder, model

def main():
    # 1. Initialize the Argument Parser
    parser = argparse.ArgumentParser(
        description="GizzBrain: Neural Net MP3 Annotation Tool for Live King Gizzard and The Lizard Wizard"
    )
    
    # 2. Define subcommands (e.g., 'gizzbrain scan' vs 'gizzbrain train')
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

    # 3. Parse the user's input
    args = parser.parse_args()

    # 4. Route the execution based on the chosen command
    if args.command == "scan":
        print(f"Scanning directory: {args.directory}...")
        mp3_files = tagger.find_files(args.directory)
        
        if not mp3_files:
            print("No MP3 files found in the specified directory.")
            return

        print(f"Found {len(mp3_files)} files. Parsing metadata...")
        
        df = tagger.convert_metadata(mp3_files)
        
        # Save results using Pandas and PyArrow
        output_path = args.output
        df_plain = pd.DataFrame(df.to_dict(orient='list'))
        df_plain.to_parquet(output_path, engine='pyarrow')
        print(f"Library saved successfully to {output_path}")

    elif args.command == "train":
        print(f"Loading metadata from {args.data}...")
        df_original = pd.read_parquet(args.data)
        
        # 1. Target the Top 10 Songs
        top_songs = df_original['title'].value_counts().head(10).index.tolist()
        print(f"Vanguard Test: Targeting Top 10 songs: {top_songs}")
        
        df_vanguard = df_original[df_original['title'].isin(top_songs)].copy()
        label_map = {title: idx for idx, title in enumerate(top_songs)}
        
        # 2. THE FILE-LEVEL SPLIT (Anti-Leakage Protocol)
        train_dfs = []
        val_dfs = []
        
        for song in top_songs:
            # Isolate all MP3s for this specific song
            song_df = df_vanguard[df_vanguard['title'] == song]
            
            # Shuffle the files
            song_df = song_df.sample(frac=1, random_state=42).reset_index(drop=True)
            
            total_files = len(song_df)
            
            # Calculate an 80/20 split, but force at least 1 file into validation 
            # if the user downloaded multiple performances of this song.
            if total_files > 1:
                split_idx = int(total_files * 0.8)
                if split_idx == total_files:
                    split_idx = total_files - 1
                elif split_idx == 0:
                    split_idx = 1
            else:
                # If only 1 recording exists across all downloaded albums, it must go to training.
                split_idx = 1
                
            train_dfs.append(song_df.iloc[:split_idx])
            val_dfs.append(song_df.iloc[split_idx:])
            
        # Recombine the separated files
        train_mp3s = pd.concat(train_dfs)
        val_mp3s = pd.concat(val_dfs) if any(len(df) > 0 for df in val_dfs) else pd.DataFrame()
        
        print(f"Split completed. Training MP3s: {len(train_mp3s)} | Validation MP3s: {len(val_mp3s)}")
        
        # 3. CHUNKING (Post-Split)
        print("Chunking training audio (this may take a few minutes)...")
        train_chunks = encoder.create_chunk_index(train_mp3s, chunk_length=5.0)
        
        print("Chunking validation audio...")
        val_chunks = encoder.create_chunk_index(val_mp3s, chunk_length=5.0)
        
        # 4. DATASETS & TRAINING
        train_dataset = encoder.GizzDataset(train_chunks, label_mapping=label_map)
        val_dataset = encoder.GizzDataset(val_chunks, label_mapping=label_map)
        
        trained_model = model.train_model(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            epochs=args.epochs,
            batch_size=args.batch_size
        )
        
        import torch
        torch.save(trained_model.state_dict(), "gizzbrain_weights.pt")
        print("Training complete. Weights saved to gizzbrain_weights.pt")

    else:
        # If the user just types 'gizzbrain' with no command, show the help menu
        parser.print_help()

if __name__ == "__main__":
    main()