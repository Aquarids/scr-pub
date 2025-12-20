from abc import ABC, abstractmethod

from utils.logger import Logger

class BaseAttack(ABC):

    def __init__(self, config, logger: Logger):
        super().__init__()
        self.config = config
        self.logger = logger
