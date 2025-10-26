from pathlib import Path


class IOMixin:
    def __init__(self, *args: str) -> None:
        self.save_dir = Path(*args)

        # make sure results folder exists
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def gen_output_path(self, filename: str) -> Path:
        return Path(self.save_dir, filename)
