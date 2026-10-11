"""Loop detection agent implementation."""

from minisweagent.agents.default import AgentConfig, DefaultAgent
from minisweagent.agents.utils.loop_detection import LoopDetectionConfig


class LoopDetectionAgentConfig(AgentConfig):
    loop_detection: LoopDetectionConfig = LoopDetectionConfig(enabled=True)


class LoopDetectionAgent(DefaultAgent):
    """DefaultAgent with loop detection enabled by default."""

    def __init__(self, *args, config_class: type = LoopDetectionAgentConfig, **kwargs):
        super().__init__(*args, config_class=config_class, **kwargs)
