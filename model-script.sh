mkdir -p /Users/Ravi.Sankar.Karuturi/tmp/hf_cache
export HF_HOME=/Users/Ravi.Sankar.Karuturi/tmp/hf_cache

# Activate your venv if using one
python3 -m venv .venv
source .venv/bin/activate
pip install colpali-engine

python -c "
from colpali_engine.models import ColPali, ColPaliProcessor;
ColPali.from_pretrained('vidore/colpali-v1.2', cache_dir='$HF_HOME', trust_remote_code=True);
ColPaliProcessor.from_pretrained('vidore/colpali-v1.2', cache_dir='$HF_HOME')"
