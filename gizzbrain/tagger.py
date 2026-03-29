# gizzbrain/tagger.py

import os
import re
import pandas as pd
from mutagen.id3 import ID3, TIT2, TPE1, TALB

def get_tags(path, verbose=False):
    """Collect tags from already annotated files."""
    tags = ID3(path)

    title = tags.get('TIT2').text[0] if tags.get('TIT2') else "Unknown Title"
    artist = tags.get('TPE1').text[0] if tags.get('TPE1') else "Unknown Artist"
    album = tags.get('TALB').text[0] if tags.get('TALB') else "Unknown Album"

    if verbose:
        print(f"Title: {title}")
        print(f"Artist: {artist}")
        print(f"Album: {album}")
        print('-' * 20)

    return {'title': title, 'artist': artist, 'album': album}

def add_tags(tag_dict, path, verbose=False):
    """Save tags stored in memory to file."""
    tags = ID3(path)

    tags.add(TIT2(encoding=3, text=tag_dict['title']))
    tags.add(TPE1(encoding=3, text=tag_dict['artist']))
    tags.add(TALB(encoding=3, text=tag_dict['album']))

    tags.save()

    if verbose:
        print(f"Tags saved to {path}")

def parse_filename(filename):
    """Parse the filename for song metadata, creates tag dictionary compatible with add_tags()."""
    pattern = r"(?P<artist>.*?)\s-\s(?P<album>.*?)\s-\s(?P<track_number>\d+)\s(?P<title>.*?)\s(?P<full_title>\(.*?\))\.mp3"
    
    match = re.search(pattern, filename)
    
    if match:
        return match.groupdict()
    else:
        print(f"Parsing Failed: {filename}")
        return {
            'artist': None, 
            'album': None, 
            'track_number': None, 
            'title': None, 
            'full_title': None
        }

def find_files(directory, extension='.mp3'):
    """Recursively finds all files of a specific type."""
    if not extension.startswith('.'):
        extension = '.' + extension
    
    found_files = []
    
    try:
        with os.scandir(directory) as entries:
            for entry in entries:
                if entry.is_file():
                    if entry.name.lower().endswith(extension.lower()):
                        found_files.append(entry.name)
                elif entry.is_dir():
                    found_files.extend(find_files(entry.path, extension))
    except PermissionError:
        pass
    
    return found_files

def convert_metadata(paths):
    """Pass a list of paths to parse for metadata and convert to a dataframe."""
    all_metadata = []
    for path in paths:
        filename = os.path.basename(path)
        metadata = parse_filename(filename)
        metadata['path'] = path
        metadata['filename'] = filename
        all_metadata.append(metadata)

    return pd.DataFrame(all_metadata)

def update_song(df, metadata, filename):
    """Update a specific row in the dataframe with manual metadata."""
    for key in metadata:
        df.loc[df['filename'] == filename, key] = metadata[key]