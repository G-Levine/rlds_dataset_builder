from typing import Iterator, Tuple, Any
import os
import json
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import imageio
from something_something_v2.conversion_utils import MultiThreadedDatasetBuilder


def _generate_examples(paths) -> Iterator[Tuple[str, Any]]:
    """Yields episodes from a list of data paths."""

    def _parse_example(item):
        """Parses an episode from a single video file."""
        video_path = item['path']
        language_instruction = item['lang']
        actions = item['actions']

        # Load all frames from the video
        video_frames = imageio.mimread(video_path)

        # Center crop each frame to a square aspect ratio
        height = video_frames[0].shape[0]
        width = video_frames[0].shape[1]
        
        # Determine the smaller dimension to crop to a square
        square_size = min(height, width)
        
        if width > square_size:
            # Center crop horizontally to square size
            video_frames = [
                frame[:, (width - square_size) // 2: (width + square_size) // 2]
                for frame in video_frames
            ]
        elif height > square_size:
            # Center crop vertically to square size
            video_frames = [
                frame[(height - square_size) // 2: (height + square_size) // 2, :]
                for frame in video_frames
            ]

        # Resize each frame to 128x128
        video_frames = [
            tf.image.resize_with_pad(frame, 128, 128).numpy().astype(np.uint8)
            for frame in video_frames
        ]

        # Initialize an empty list to collect the steps
        steps = []

        for i, frame in enumerate(video_frames):
            # Construct the action vector
            action_vector = np.zeros((16,), dtype=np.float32) if not actions else np.array(actions[i])

            # Create a dictionary representing a single step
            step_data = {
                'observation': {
                    'image': frame
                },
                'action': action_vector,
                'language_instruction': language_instruction,
            }

            steps.append(step_data)

        # Create the episode with all steps included
        return video_path, {
            'steps': steps,
            'episode_metadata': {'file_path': video_path}
        }

    for item in paths:
        yield _parse_example(item)


class SomethingSomethingV2Dataset(MultiThreadedDatasetBuilder):
    """Multithreaded DatasetBuilder for the example dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
    }
    N_WORKERS = 200             # Number of parallel workers for data conversion
    MAX_PATHS_IN_MEMORY = 1000  # Number of paths converted & stored in memory before writing to disk
                               # Adjust based on available RAM
    PARSE_FCN = _generate_examples  # Function handle for parsing file paths to RLDS episodes

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation, etc.)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'image': tfds.features.Image(
                            shape=(128, 128, 3),
                            dtype=np.uint8,
                            encoding_format='jpeg',
                            doc='Main camera RGB observation.',
                        ),
                    }),
                    'action': tfds.features.Tensor(
                        shape=(16,),
                        dtype=np.float32,
                        doc='Latent actions.',
                    ),
                    'language_instruction': tfds.features.Text(
                        doc='Language Instruction.'
                    ),
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(
                        doc='Path to the original data file.'
                    ),
                }),
            }))

    def _split_paths(self):
        """Define file paths for data splits."""
        base_path = os.path.expanduser('~/20bn-something-something-v2/')
        labels_path = os.path.expanduser('~/labels/')
        actions_path = os.path.expanduser('~/')
        filter_labels_path = os.path.join(os.path.dirname(__file__), 'labels_filtered.json')

        with open(filter_labels_path, "r") as f:
            filter_labels = set(json.load(f).keys())
        
        with open(os.path.join(labels_path, 'train.json'), 'r') as f:
            train_annotations = json.load(f)
        with open(os.path.join(labels_path, 'validation.json'), 'r') as f:
            val_annotations = json.load(f)
        train_actions = None
        val_actions = None
        try:
            with open(os.path.join(actions_path, 'train_actions.json'), 'r') as f:
                train_actions = json.load(f)
            with open(os.path.join(actions_path, 'val_actions.json'), 'r') as f:
                val_actions = json.load(f)
        except FileNotFoundError:
            pass
            
        train_annotations = [
            x
            for x in train_annotations
            if x["template"].replace("[", "").replace("]", "") in filter_labels
        ]
        val_annotations = [
            x
            for x in val_annotations
            if x["template"].replace("[", "").replace("]", "") in filter_labels
        ]

        # Format to have paths and labels
        train_paths = [
            {'path': os.path.join(base_path, f"{x['id']}.webm"), 'lang': x['label'], 'actions': None if not train_actions else train_actions[x['id']]}
            for x in train_annotations
        ]
        val_paths = [
            {'path': os.path.join(base_path, f"{x['id']}.webm"), 'lang': x['label'], 'actions': None if not val_actions else val_actions[x['id']]}
            for x in val_annotations
        ]

        return {
            'train': train_paths,
            'val': val_paths,
        }
