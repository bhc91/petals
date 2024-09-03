from petals.models.mamba.block import WrappedMambaBlock
from petals.models.mamba.config import DistributedMambaConfig
from petals.models.mamba.model import (
    DistributedMambaForCausalLM,
    DistributedMambaForSequenceClassification,
    DistributedMambaModel,
)
from petals.models.mamba.speculative_model import DistributedMambaForSpeculativeGeneration
from petals.utils.auto_config import register_model_classes

register_model_classes(
    config=DistributedMambaConfig,
    model=DistributedMambaModel,
    model_for_causal_lm=DistributedMambaForCausalLM,
    model_for_speculative=DistributedMambaForSpeculativeGeneration,
    model_for_sequence_classification=DistributedMambaForSequenceClassification,
)
