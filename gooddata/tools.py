import os
from pathlib import Path
from urllib.parse import urlparse

TMP_DIR = Path("tmp")


def get_name_for_id(objects, object_id: str) -> str:
    for o in objects:
        if o.id == object_id:
            return o.name


def get_title_for_id(objects, object_id: str) -> str:
    for o in objects:
        if o.id == object_id:
            return o.title


def create_dir(file_dir: Path) -> None:
    if not os.path.isdir(file_dir):
        os.mkdir(file_dir)


def get_org_id_from_host(gd_url: str) -> str:
    return urlparse(gd_url).netloc.split(".")[0]
