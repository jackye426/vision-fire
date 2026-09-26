"""Download and verify PyroNear's packaged v0.4.0 temporal smoke model."""

import hashlib
from pathlib import Path

from huggingface_hub import hf_hub_download


ROOT = Path(__file__).resolve().parent
REVISION = "8b97c9f497c8cdfbda38b2bab21a00485ee8671e"
SHA256 = "0fb060ca81a6e36944f28f80e81b1ae9e5b8557b207aca8a600fb9635547cb28"
TARGET = ROOT / "models" / "pyronear_temporal" / "model.zip"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    if not TARGET.exists():
        TARGET.parent.mkdir(parents=True, exist_ok=True)
        hf_hub_download("pyronear/temporal-model", "model.zip", revision=REVISION,
                        local_dir=str(TARGET.parent))
    actual = file_sha256(TARGET)
    if actual != SHA256:
        raise ValueError(f"Unexpected PyroNear model SHA-256: {actual}")
    print(f"Verified {TARGET} ({TARGET.stat().st_size:,} bytes)")
