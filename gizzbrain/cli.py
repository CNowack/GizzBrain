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
        
        # Build full paths for the tagger
        # Note: find_files might return just filenames depending on OS walk, 
        # so we ensure absolute paths are passed to the converter.
        full_paths = [os.path.join(args.directory, f) for f in mp3_files]
        df = tagger.convert_metadata(full_paths)
        
        # Save results using Pandas and PyArrow
        output_path = args.output
        df_plain = pd.DataFrame(df.to_dict(orient='list'))
        df_plain.to_parquet(output_path, engine='pyarrow')
        print(f"Library saved successfully to {output_path}")

    elif args.command == "train":
        print(f"Loading metadata from {args.data}...")
        df_original = pd.read_parquet(args.data)
        
        print("Chunking audio indexing (5-second intervals)...")
        df_chunks = encoder.create_chunk_index(df_original, chunk_length=5.0)
        
        print(f"Generated {len(df_chunks)} training chunks. Initializing dataset...")
        dataset = encoder.GizzDataset(df_chunks)
        
        # Pass the dataset to the model's training loop
        trained_model = model.train_model(
            dataset=dataset, 
            epochs=args.epochs, 
            batch_size=args.batch_size
        )
        
        print("Training complete. (Model saving logic to be implemented).")

    else:
        # If the user just types 'gizzbrain' with no command, show the help menu
        parser.print_help()

if __name__ == "__main__":
    main()