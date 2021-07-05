from pathlib import Path
from typing import List, Union


def sglob(
        path: Union[Path, str],
        pattern: str = '**/*'
        ) -> List[Path]:
    path = Path(path)
    return sorted(list(path.glob(pattern)))
