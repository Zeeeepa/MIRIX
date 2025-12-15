"""
In-memory queue implementation using Python's queue.Queue
Thread-safe and suitable for single-process applications

Includes PartitionedMemoryQueue for simulating Kafka-like partitioning
where messages are routed by user_id to ensure per-user ordering.
"""

import logging
import queue
from typing import List, Optional

from mirix.queue.message_pb2 import QueueMessage
from mirix.queue.queue_interface import QueueInterface

logger = logging.getLogger(__name__)


class MemoryQueue(QueueInterface):
    """Thread-safe in-memory queue implementation (single partition)"""

    def __init__(self):
        """Initialize the in-memory queue"""
        self._queue = queue.Queue()

    def put(self, message: QueueMessage) -> None:
        """
        Add a message to the in-memory queue

        Args:
            message: QueueMessage protobuf message to add
        """
        logger.debug("Adding message to queue: agent_id=%s", message.agent_id)
        self._queue.put(message)

    def get(self, timeout: Optional[float] = None) -> QueueMessage:
        """
        Retrieve a message from the queue

        Args:
            timeout: Optional timeout in seconds (None = block indefinitely)

        Returns:
            QueueMessage protobuf message from the queue

        Raises:
            queue.Empty: If no message available within timeout
        """
        message = self._queue.get(timeout=timeout)
        logger.debug("Retrieved message from queue: agent_id=%s", message.agent_id)
        return message

    def close(self) -> None:
        """
        Clean up resources
        For in-memory queue, no cleanup is needed
        """
        pass


class PartitionedMemoryQueue(QueueInterface):
    """
    Partitioned in-memory queue that simulates Kafka's partitioning behavior.

    Messages are routed to partitions based on user_id hash, ensuring:
    - All messages for the same user go to the same partition
    - Each partition is consumed by exactly one worker
    - Per-user message ordering is preserved (same as Kafka behavior)

    This allows parallel processing across users while maintaining
    serial processing within each user's message stream.
    """

    def __init__(self, num_partitions: int = 1):
        """
        Initialize the partitioned queue

        Args:
            num_partitions: Number of partitions (should match NUM_WORKERS)
        """
        self._num_partitions = max(1, num_partitions)
        self._partitions: List[queue.Queue] = [
            queue.Queue() for _ in range(self._num_partitions)
        ]
        logger.info(
            "Initialized PartitionedMemoryQueue with %d partitions",
            self._num_partitions,
        )

    @property
    def num_partitions(self) -> int:
        """Get the number of partitions"""
        return self._num_partitions

    def _get_partition_key(self, message: QueueMessage) -> str:
        """
        Extract partition key from message (mirrors KafkaQueue behavior)

        Args:
            message: QueueMessage to extract key from

        Returns:
            Partition key string (user_id or actor.id as fallback)
        """
        # Match KafkaQueue's partition key logic from kafka_queue.py
        if message.HasField("user_id") and message.user_id:
            return message.user_id
        elif message.HasField("actor") and message.actor.id:
            return message.actor.id
        else:
            # Fallback to agent_id if no user context
            return message.agent_id

    def _compute_partition(self, partition_key: str) -> int:
        """
        Compute partition number from key (mirrors Kafka's partitioning)

        Uses hash(key) % num_partitions, same as Kafka's default partitioner.

        Args:
            partition_key: Key to hash

        Returns:
            Partition index (0 to num_partitions-1)
        """
        return hash(partition_key) % self._num_partitions

    def put(self, message: QueueMessage) -> None:
        """
        Add a message to the appropriate partition based on user_id

        Args:
            message: QueueMessage protobuf message to add
        """
        partition_key = self._get_partition_key(message)
        partition_id = self._compute_partition(partition_key)

        logger.debug(
            "Routing message to partition %d: agent_id=%s, partition_key=%s",
            partition_id,
            message.agent_id,
            partition_key,
        )

        self._partitions[partition_id].put(message)

    def get(self, timeout: Optional[float] = None) -> QueueMessage:
        """
        Retrieve a message from partition 0 (for backward compatibility)

        Note: For partitioned queues, use get_from_partition() instead.

        Args:
            timeout: Optional timeout in seconds

        Returns:
            QueueMessage from partition 0

        Raises:
            queue.Empty: If no message available within timeout
        """
        return self.get_from_partition(0, timeout)

    def get_from_partition(
        self, partition_id: int, timeout: Optional[float] = None
    ) -> QueueMessage:
        """
        Retrieve a message from a specific partition

        Args:
            partition_id: Partition to consume from (0 to num_partitions-1)
            timeout: Optional timeout in seconds (None = block indefinitely)

        Returns:
            QueueMessage protobuf message from the specified partition

        Raises:
            queue.Empty: If no message available within timeout
            ValueError: If partition_id is out of range
        """
        if partition_id < 0 or partition_id >= self._num_partitions:
            raise ValueError(
                f"Invalid partition_id {partition_id}, "
                f"must be 0 to {self._num_partitions - 1}"
            )

        message = self._partitions[partition_id].get(timeout=timeout)
        logger.debug(
            "Retrieved message from partition %d: agent_id=%s",
            partition_id,
            message.agent_id,
        )
        return message

    def close(self) -> None:
        """
        Clean up resources
        For in-memory queue, no cleanup is needed
        """
        pass
