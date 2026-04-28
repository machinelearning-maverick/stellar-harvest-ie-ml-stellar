from typing import List

import pandas as pd

from stellar_harvest_ie_config.utils.log_decorators import log_io
from stellar_harvest_ie_store.db import AsyncSessionLocal
from stellar_harvest_ie_store.repository import AsyncRepository
from stellar_harvest_ie_models.stellar.swpc.entities import KpIndexEntity


def kp_entities_to_df(entities: List[KpIndexEntity]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "id": e.id,
                "time_tag": e.time_tag,
                "kp_index": e.kp_index,
                "estimated_kp": e.estimated_kp,
                "kp": e.kp,
            }
            for e in entities
        ]
    )


@log_io(
    skip_types_input={
        pd.DataFrame: lambda v: f"<DataFrame shape={v.shape} columns={list(v.columns)}>",
    }
)
async def load_planetary_kp_index() -> pd.DataFrame:
    async with AsyncSessionLocal() as session:
        repository = AsyncRepository(KpIndexEntity, session)
        indices: List[KpIndexEntity] = await repository.list()
        return kp_entities_to_df(indices)
