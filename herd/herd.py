import json

class HerdModel:
    def __init__(self):
        self.expert = None
        pass


class Experts:
    def __init__(self, experts):
        self.experts = experts

    @staticmethod
    def parse_file(expert_file: str = 'experts.json'):
        with open(expert_file, 'r') as json_file:
            # Parse the JSON data from the file
            return Experts(**json.loads(json_file))


class Expert:
    def __init__(self):
        self.name
        self.descriptions
