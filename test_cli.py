from pathlib import Path
from arka.config.loader import ConfigLoader
loader = ConfigLoader()
config = loader.load(Path("examples/08-privacy-guardrails.yaml"))
print(config)
