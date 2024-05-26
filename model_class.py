from abc import ABC, abstractmethod
from typing import Iterable


class Model(ABC):
    """Model that can take in 1 producer and give id of 10 consumers that are most likely to consume the energy produced by the producer. And take in 1 consumer and give id of 10 producers that are most likely to produce the energy consumed by the consumer."""

    @abstractmethod
    def predict_consumers(self, producer_id: int) -> Iterable[tuple[int, float]]:
        """Predict the 10 consumers that are most likely to consume the energy produced by the producer."""
        # Throw not implemented error
        raise NotImplementedError

    @abstractmethod
    def predict_producers(self, consumer_id: int) -> Iterable[tuple[int, float]]:
        """Predict the 10 producers that are most likely to produce the energy consumed by the consumer."""
        # Throw not implemented error
        raise NotImplementedError