from pydantic_settings import BaseSettings
from pathlib import Path
from typing import Iterator


class Settings(BaseSettings):
    data_dir: Path = Path("data")
    train_dir: Path = data_dir / "train"
    test_dir: Path = data_dir / "test"

    def iterate_all_train_images(self) -> Iterator[tuple[str, str, Path]]:
        for subdir in self.train_dir.iterdir():
            label = subdir.name
            for image_dir in subdir.iterdir():
                for image_path in _iterate_image_paths(image_dir):
                    yield label, image_dir.name, image_path

    def iterate_all_test_images(self) -> Iterator[tuple[str, str, Path]]:
        for subdir in self.test_dir.iterdir():
            label = subdir.name
            for image_dir in subdir.iterdir():
                for image_path in _iterate_image_paths(image_dir):
                    yield label, image_dir.name, image_path


def _iterate_image_paths(dir_path: Path) -> Iterator[Path]:
    for image_path in dir_path.iterdir():
        if image_path.suffix.lower() in {".jpg", ".jpeg", ".webp", ".png"}:
            yield image_path
