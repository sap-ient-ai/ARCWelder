from pathlib import Path
import json

import setup_ipynb

from Puzzle import Puzzle

class ARC:
    LOOKUP = None

    @staticmethod
    def _locate_arc_data_dir():
        L = [ ancestor / 'ARC-AGI' / 'data'
             for ancestor in reversed(Path(__file__).parents)
        ]
        return next((u for u in L if u.is_dir()), None)

    @staticmethod
    def _generate_lookup():
        puzzle_filenames = lambda dataset: [ f.name.rstrip('.json')
            for f in (ARC._locate_arc_data_dir() / dataset).iterdir()
                if f.is_file()
        ]
        return {
            dataset: sorted(puzzle_filenames(dataset))
                for dataset in ['training', 'evaluation']
        }

    @staticmethod
    def _lookup():
        if ARC.LOOKUP is None:
            ARC.LOOKUP = ARC._generate_lookup()
        return ARC.LOOKUP

    @staticmethod
    def find_from(hex_=None):
        hex_ = hex_.rstrip('.json')
        for dataset in ['training', 'evaluation']:
            if hex_ in ARC._lookup()[dataset]:
                return dataset, ARC._lookup()[dataset].index(hex_)
        raise ValueError(f'Puzzle not found for {hex_}')

    @staticmethod
    def load(*args, **kwargs):
        dataset = kwargs.get('dataset', 'training')
        index = kwargs.get('index', None)
        hex_ = kwargs.get('hex', None)

        if len(args) == 1:
            u = args[0]
            if isinstance(u, int):  index = u
            if isinstance(u, str):  hex_ = u

        if hex_ is not None:
            dataset, index = ARC.find_from(hex_)
            assert dataset is not None and index is not None,  \
                "No dataset or index found from hex code"

        if dataset is None or index is None:
            raise ValueError('Supply either dataset + index or hex')

        file_path = (
            ARC._locate_arc_data_dir() / dataset / ARC._lookup()[dataset][index]
        ).with_suffix('.json')

        with open(file_path, 'r') as f:
            J = json.load(f)

        return Puzzle(J)
