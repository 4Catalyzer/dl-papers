from datetime import datetime
import os

# -----------------------------------------------------------------------------

# dl-papers/dl_papers/env.py
ROOT_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__),
    ),
)

DATA_DIR = os.path.join(ROOT_DIR, 'data')

RUN_NAME = os.environ.get('RUN_NAME') or 'fit'
RUN_ID = os.environ.get('RUN_ID') or datetime.now().strftime('%Y%m%d-%H%M%S')

OUTPUT_DIR = os.path.join(ROOT_DIR, 'output', RUN_NAME, RUN_ID)
