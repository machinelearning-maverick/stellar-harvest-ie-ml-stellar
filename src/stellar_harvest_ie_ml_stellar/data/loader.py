from typing import List

import pandas as pd

from stellar_harvest_ie_config.utils.log_decorators import log_io
from stellar_harvest_ie_store.db import AsyncSessionLocal
from stellar_harvest_ie_store.repository import AsyncRepository
from stellar_harvest_ie_models.stellar.swpc.entities import KpIndexEntity


@log_io()
def load_planetary_kp_index(url: str) -> pd.DataFrame:
    return pd.read_json(url)


@log_io()
async def load_planetary_kp_index() -> pd.DataFrame:
    async with AsyncSessionLocal() as session:
        repository = AsyncRepository(KpIndexEntity, session)
        indices: List[KpIndexEntity] = repository.list()
        df = pd.DataFrame(
            [
                {
                    "id": e.id,
                    "time_tag": e.time_tag,
                    "kp_index": e.kp_index,
                    "estimated_kp": e.estimated_kp,
                    "kp": e.kp,
                }
                for e in indices
            ]
        )
        return df
