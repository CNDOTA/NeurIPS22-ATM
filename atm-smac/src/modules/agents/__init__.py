REGISTRY = {}

from .rnn_agent import RNNAgent
from .memory_agent import MemoryAgent

REGISTRY["rnn"] = RNNAgent
REGISTRY["memory"] = MemoryAgent