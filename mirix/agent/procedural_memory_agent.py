from mirix.agent.agent import Agent


class ProceduralMemoryAgent(Agent):
    def __init__(self, **kwargs):
        # load parent class init
        super().__init__(**kwargs)
