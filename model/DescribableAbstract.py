class DescribableAbstract:
    def __init__(self) -> None:
        self.description: str = None

    def set_description(self, description: str):
        self.description = description
        self.record["document"] = self.description
