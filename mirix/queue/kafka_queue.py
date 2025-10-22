"""
Kafka queue implementation
Requires kafka-python and protobuf libraries to be installed
Uses Google Protocol Buffers for message serialization
"""
import logging
from typing import Optional

from mirix.queue.queue_interface import QueueInterface
from mirix.queue.message_pb2 import QueueMessage

logger = logging.getLogger(__name__)


class KafkaQueue(QueueInterface):
    """Kafka-based queue implementation using Protocol Buffers"""
    
    def __init__(self, bootstrap_servers: str, topic: str, group_id: str):
        """
        Initialize Kafka producer and consumer with Protobuf serialization
        
        Args:
            bootstrap_servers: Kafka broker address(es)
            topic: Kafka topic name
            group_id: Consumer group ID
        """
        logger.debug(f"Initializing Kafka queue: servers={bootstrap_servers}, topic={topic}, group={group_id}")
        
        try:
            from kafka import KafkaProducer, KafkaConsumer
        except ImportError:
            logger.error("kafka-python not installed")
            raise ImportError(
                "kafka-python is required for Kafka support. "
                "Install it with: pip install queue-sample[kafka]"
            )
        
        self.topic = topic
        
        # Protobuf serializer: Convert QueueMessage to bytes
        def protobuf_serializer(message: QueueMessage) -> bytes:
            """
            Serialize QueueMessage to Protocol Buffer format
            
            Args:
                message: QueueMessage protobuf to serialize
                
            Returns:
                Serialized protobuf bytes
            """
            return message.SerializeToString()
        
        # Protobuf deserializer: Convert bytes to QueueMessage
        def protobuf_deserializer(serialized_msg: bytes) -> QueueMessage:
            """
            Deserialize Protocol Buffer message to QueueMessage
            
            Args:
                serialized_msg: Serialized protobuf bytes
                
            Returns:
                QueueMessage protobuf object
            """
            msg = QueueMessage()
            msg.ParseFromString(serialized_msg)
            return msg
        
        # Initialize Kafka producer with Protobuf serializer
        self.producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=protobuf_serializer
        )
        
        # Initialize Kafka consumer with Protobuf deserializer
        self.consumer = KafkaConsumer(
            topic,
            bootstrap_servers=bootstrap_servers,
            group_id=group_id,
            value_deserializer=protobuf_deserializer,
            auto_offset_reset='earliest',  # Start from beginning if no offset exists
            enable_auto_commit=True,
            consumer_timeout_ms=1000  # Timeout for polling
        )
    
    def put(self, message: QueueMessage) -> None:
        """
        Send a message to Kafka topic
        
        Args:
            message: QueueMessage protobuf message to send
        """
        logger.debug(f"Sending message to Kafka topic {self.topic}: agent_id={message.agent_id}")
        
        # Send message and wait for acknowledgment
        future = self.producer.send(self.topic, value=message)
        future.get(timeout=10)  # Wait up to 10 seconds for confirmation
        
        logger.debug(f"Message sent to Kafka successfully")
    
    def get(self, timeout: Optional[float] = None) -> QueueMessage:
        """
        Retrieve a message from Kafka
        
        Args:
            timeout: Not used for Kafka (uses consumer_timeout_ms instead)
            
        Returns:
            QueueMessage protobuf message from Kafka
            
        Raises:
            StopIteration: If no message available
        """
        logger.debug(f"Polling Kafka topic {self.topic} for messages")
        
        # Poll for messages
        for message in self.consumer:
            logger.debug(f"Retrieved message from Kafka: agent_id={message.value.agent_id}")
            return message.value
        
        # If no message received, raise exception (similar to queue.Empty)
        logger.debug("No message available from Kafka")
        raise StopIteration("No message available")
    
    def close(self) -> None:
        """Close Kafka producer and consumer connections"""
        logger.info("Closing Kafka connections")
        
        if hasattr(self, 'producer'):
            self.producer.close()
            logger.debug("Kafka producer closed")
        if hasattr(self, 'consumer'):
            self.consumer.close()
            logger.debug("Kafka consumer closed")

