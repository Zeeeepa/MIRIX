"""
Unit tests for Mirix Queue System

Tests cover:
- Queue initialization and lifecycle
- Message enqueueing and processing
- Worker thread management
- Queue manager functionality
- Memory queue implementation
"""

import pytest
import time
import threading
from unittest.mock import Mock
from datetime import datetime

# Import queue components
from mirix.queue import initialize_queue, save
from mirix.queue.manager import get_manager
from mirix.queue.worker import QueueWorker
from mirix.queue.memory_queue import MemoryQueue
from mirix.queue.queue_util import put_messages
# Note: ProtoUser and ProtoMessageCreate are generated from message.proto
from mirix.queue.message_pb2 import QueueMessage, User as ProtoUser, MessageCreate as ProtoMessageCreate

# Import schemas
from mirix.schemas.client import Client
from mirix.schemas.message import MessageCreate
from mirix.schemas.enums import MessageRole


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_server():
    """Create a mock SyncServer instance"""
    server = Mock()
    server.send_messages = Mock(return_value=Mock(
        model_dump=Mock(return_value={
            "completion_tokens": 100,
            "prompt_tokens": 50
        })
    ))
    return server


@pytest.fixture
def sample_client():
    """Create a sample Client (represents a client application)"""
    return Client(
        id="client-123",
        organization_id="org-456",
        name="Test Client App",
        status="active",
        scope="read_write",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        is_deleted=False
    )


@pytest.fixture
def sample_messages():
    """Create sample MessageCreate list"""
    return [
        MessageCreate(
            role=MessageRole.user,
            content="Hello, how are you?"
        ),
        MessageCreate(
            role=MessageRole.user,
            content="What's the weather like?"
        )
    ]


@pytest.fixture
def sample_queue_message(sample_client):
    """Create a sample QueueMessage protobuf"""
    msg = QueueMessage()
    
    # Set actor (Client converted to protobuf User)
    msg.actor.id = sample_client.id
    msg.actor.organization_id = sample_client.organization_id
    msg.actor.name = sample_client.name
    msg.actor.status = sample_client.status
    msg.actor.timezone = "UTC"  # Client doesn't have timezone, use default
    msg.actor.is_deleted = sample_client.is_deleted
    
    # Set agent and messages
    msg.agent_id = "agent-789"
    
    proto_msg = ProtoMessageCreate()
    proto_msg.role = ProtoMessageCreate.ROLE_USER
    proto_msg.text_content = "Test message"
    msg.input_messages.append(proto_msg)
    
    msg.chaining = True
    msg.verbose = False
    
    return msg


@pytest.fixture
def clean_manager():
    """Get a fresh QueueManager for testing"""
    # Get the singleton and cleanup if initialized
    manager = get_manager()
    if manager.is_initialized:
        manager.cleanup()
    return manager


# ============================================================================
# MemoryQueue Tests
# ============================================================================

class TestMemoryQueue:
    """Test the in-memory queue implementation"""
    
    def test_memory_queue_init(self):
        """Test MemoryQueue initialization"""
        queue = MemoryQueue()
        assert queue is not None
        assert hasattr(queue, '_queue')
    
    def test_memory_queue_put_get(self, sample_queue_message):
        """Test putting and getting messages from memory queue"""
        queue = MemoryQueue()
        
        # Put message
        queue.put(sample_queue_message)
        
        # Get message
        retrieved = queue.get(timeout=1.0)
        
        assert retrieved.agent_id == sample_queue_message.agent_id
        assert retrieved.actor.id == sample_queue_message.actor.id
        assert len(retrieved.input_messages) == len(sample_queue_message.input_messages)
    
    def test_memory_queue_timeout(self):
        """Test that get() raises Empty on timeout"""
        import queue as q
        
        mem_queue = MemoryQueue()
        
        with pytest.raises(q.Empty):
            mem_queue.get(timeout=0.1)
    
    def test_memory_queue_fifo_order(self, sample_client):
        """Test that messages are retrieved in FIFO order"""
        queue = MemoryQueue()
        
        # Create multiple messages
        messages = []
        for i in range(5):
            msg = QueueMessage()
            msg.actor.id = sample_client.id
            msg.agent_id = f"agent-{i}"
            messages.append(msg)
            queue.put(msg)
        
        # Retrieve and verify order
        for i in range(5):
            retrieved = queue.get(timeout=1.0)
            assert retrieved.agent_id == f"agent-{i}"
    
    def test_memory_queue_close(self):
        """Test queue close (should not raise error)"""
        queue = MemoryQueue()
        queue.close()  # Should complete without error


# ============================================================================
# QueueWorker Tests
# ============================================================================

class TestQueueWorker:
    """Test the queue worker functionality"""
    
    def test_worker_init_without_server(self):
        """Test worker initialization without server"""
        queue = MemoryQueue()
        worker = QueueWorker(queue)
        
        assert worker.queue == queue
        assert worker._server is None
        assert worker._running is False
    
    def test_worker_init_with_server(self, mock_server):
        """Test worker initialization with server"""
        queue = MemoryQueue()
        worker = QueueWorker(queue, server=mock_server)
        
        assert worker.queue == queue
        assert worker._server == mock_server
    
    def test_worker_set_server(self, mock_server):
        """Test setting server after initialization"""
        queue = MemoryQueue()
        worker = QueueWorker(queue)
        
        assert worker._server is None
        
        worker.set_server(mock_server)
        
        assert worker._server == mock_server
    
    def test_worker_start_stop(self):
        """Test worker thread lifecycle"""
        queue = MemoryQueue()
        worker = QueueWorker(queue)
        
        # Start worker
        worker.start()
        assert worker._running is True
        assert worker._thread is not None
        assert worker._thread.is_alive()
        
        # Stop worker
        worker.stop()
        assert worker._running is False
        
        # Wait a bit for thread to finish
        time.sleep(0.5)
        assert not worker._thread.is_alive()
    
    def test_worker_process_message_without_server(self, sample_queue_message):
        """Test processing message when no server is available"""
        queue = MemoryQueue()
        worker = QueueWorker(queue)
        
        # Should log warning and skip processing
        worker._process_message(sample_queue_message)
        # No error should be raised
    
    def test_worker_process_message_with_server(self, mock_server, sample_queue_message):
        """Test processing message with server available"""
        queue = MemoryQueue()
        worker = QueueWorker(queue, server=mock_server)
        
        # Process message
        worker._process_message(sample_queue_message)
        
        # Verify server.send_messages was called
        mock_server.send_messages.assert_called_once()
        
        # Check call arguments
        call_args = mock_server.send_messages.call_args
        assert call_args.kwargs['agent_id'] == sample_queue_message.agent_id
        assert len(call_args.kwargs['input_messages']) > 0
    
    def test_worker_message_processing_integration(self, mock_server, sample_queue_message):
        """Test end-to-end message processing"""
        queue = MemoryQueue()
        worker = QueueWorker(queue, server=mock_server)
        
        # Enqueue message
        queue.put(sample_queue_message)
        
        # Start worker
        worker.start()
        
        # Wait for processing
        time.sleep(1.0)
        
        # Stop worker
        worker.stop()
        
        # Verify server was called
        assert mock_server.send_messages.call_count >= 1


# ============================================================================
# QueueManager Tests
# ============================================================================

class TestQueueManager:
    """Test the queue manager functionality"""
    
    def test_manager_singleton(self):
        """Test that get_manager returns singleton"""
        manager1 = get_manager()
        manager2 = get_manager()
        
        assert manager1 is manager2
    
    def test_manager_init_without_server(self, clean_manager):
        """Test manager initialization without server"""
        manager = clean_manager
        
        assert not manager.is_initialized
        
        manager.initialize()
        
        assert manager.is_initialized
        assert manager._queue is not None
        assert manager._worker is not None
        
        # Cleanup
        manager.cleanup()
    
    def test_manager_init_with_server(self, clean_manager, mock_server):
        """Test manager initialization with server"""
        manager = clean_manager
        
        manager.initialize(server=mock_server)
        
        assert manager.is_initialized
        assert manager._server == mock_server
        assert manager._worker._server == mock_server
        
        # Cleanup
        manager.cleanup()
    
    def test_manager_idempotent_init(self, clean_manager, mock_server):
        """Test that multiple initialize calls are idempotent"""
        manager = clean_manager
        
        manager.initialize(server=mock_server)
        first_queue = manager._queue
        first_worker = manager._worker
        
        # Call initialize again
        manager.initialize(server=mock_server)
        
        # Should be same instances
        assert manager._queue is first_queue
        assert manager._worker is first_worker
        
        # Cleanup
        manager.cleanup()
    
    def test_manager_update_server_after_init(self, clean_manager, mock_server):
        """Test updating server after initialization"""
        manager = clean_manager
        
        # Initialize without server
        manager.initialize()
        assert manager._server is None
        
        # Update with server
        mock_server_2 = Mock()
        manager.initialize(server=mock_server_2)
        
        assert manager._server == mock_server_2
        assert manager._worker._server == mock_server_2
        
        # Cleanup
        manager.cleanup()
    
    def test_manager_save_message(self, clean_manager, sample_queue_message):
        """Test saving message via manager"""
        manager = clean_manager
        manager.initialize()
        
        # Save message
        manager.save(sample_queue_message)
        
        # Should be in queue
        retrieved = manager._queue.get(timeout=1.0)
        assert retrieved.agent_id == sample_queue_message.agent_id
        
        # Cleanup
        manager.cleanup()
    
    def test_manager_cleanup(self, clean_manager):
        """Test manager cleanup"""
        manager = clean_manager
        manager.initialize()
        
        assert manager.is_initialized
        assert manager._worker._running
        
        manager.cleanup()
        
        assert not manager.is_initialized
        assert manager._queue is None
        assert manager._worker is None


# ============================================================================
# queue_util Tests
# ============================================================================

class TestQueueUtil:
    """Test queue utility functions"""
    
    def test_put_messages_basic(self, clean_manager, sample_client, sample_messages):
        """Test put_messages with basic parameters"""
        manager = clean_manager
        manager.initialize()
        
        put_messages(
            actor=sample_client,
            agent_id="agent-789",
            input_messages=sample_messages
        )
        
        # Retrieve and verify
        msg = manager._queue.get(timeout=1.0)
        assert msg.agent_id == "agent-789"
        assert msg.actor.id == sample_client.id
        assert len(msg.input_messages) == len(sample_messages)
        
        # Cleanup
        manager.cleanup()
    
    def test_put_messages_with_options(self, clean_manager, sample_client, sample_messages):
        """Test put_messages with optional parameters"""
        manager = clean_manager
        manager.initialize()
        
        put_messages(
            actor=sample_client,
            agent_id="agent-789",
            input_messages=sample_messages,
            chaining=False,
            user_id="user-custom",
            verbose=True,
            filter_tags={"tag1": "value1"}
        )
        
        # Retrieve and verify
        msg = manager._queue.get(timeout=1.0)
        assert msg.chaining is False
        assert msg.user_id == "user-custom"
        assert msg.verbose is True
        assert dict(msg.filter_tags) == {"tag1": "value1"}
        
        # Cleanup
        manager.cleanup()
    
    def test_put_messages_role_mapping(self, clean_manager, sample_client):
        """Test that message roles are correctly mapped"""
        manager = clean_manager
        manager.initialize()
        
        messages = [
            MessageCreate(role=MessageRole.user, content="User message"),
            MessageCreate(role=MessageRole.system, content="System message"),
        ]
        
        put_messages(
            actor=sample_client,
            agent_id="agent-789",
            input_messages=messages
        )
        
        # Retrieve and verify
        msg = manager._queue.get(timeout=1.0)
        assert msg.input_messages[0].role == ProtoMessageCreate.ROLE_USER
        assert msg.input_messages[1].role == ProtoMessageCreate.ROLE_SYSTEM
        
        # Cleanup
        manager.cleanup()


# ============================================================================
# Queue __init__ Tests
# ============================================================================

class TestQueueInit:
    """Test queue module initialization functions"""
    
    def test_initialize_queue(self, clean_manager, mock_server):
        """Test initialize_queue function"""
        manager = clean_manager
        
        initialize_queue(mock_server)
        
        assert manager.is_initialized
        assert manager._server == mock_server
        
        # Cleanup
        manager.cleanup()
    
    def test_save_without_init(self, clean_manager, sample_queue_message):
        """Test save auto-initializes if not initialized"""
        manager = clean_manager
        
        assert not manager.is_initialized
        
        save(sample_queue_message)
        
        # Should auto-initialize
        assert manager.is_initialized
        
        # Cleanup
        manager.cleanup()
    
    def test_save_with_init(self, clean_manager, mock_server, sample_queue_message):
        """Test save after manual initialization"""
        manager = clean_manager
        initialize_queue(mock_server)
        
        save(sample_queue_message)
        
        # Should be in queue
        retrieved = manager._queue.get(timeout=1.0)
        assert retrieved.agent_id == sample_queue_message.agent_id
        
        # Cleanup
        manager.cleanup()


# ============================================================================
# Integration Tests
# ============================================================================

class TestQueueIntegration:
    """Integration tests for the complete queue system"""
    
    def test_end_to_end_message_flow(self, clean_manager, mock_server, sample_client, sample_messages):
        """Test complete message flow from enqueue to processing"""
        manager = clean_manager
        
        # Initialize with server
        initialize_queue(mock_server)
        
        # Enqueue message
        put_messages(
            actor=sample_client,
            agent_id="agent-integration",
            input_messages=sample_messages
        )
        
        # Wait for worker to process
        time.sleep(1.5)
        
        # Verify server was called
        assert mock_server.send_messages.call_count >= 1
        
        # Verify call details
        call_args = mock_server.send_messages.call_args
        assert call_args.kwargs['agent_id'] == "agent-integration"
        
        # Cleanup
        manager.cleanup()
    
    def test_multiple_messages_processing(self, clean_manager, mock_server, sample_client, sample_messages):
        """Test processing multiple messages"""
        manager = clean_manager
        initialize_queue(mock_server)
        
        # Enqueue multiple messages
        for i in range(5):
            put_messages(
                actor=sample_client,
                agent_id=f"agent-{i}",
                input_messages=sample_messages
            )
        
        # Wait for processing
        time.sleep(2.0)
        
        # Verify all were processed
        assert mock_server.send_messages.call_count >= 5
        
        # Cleanup
        manager.cleanup()
    
    def test_worker_handles_processing_errors(self, clean_manager, sample_client, sample_messages):
        """Test that worker handles errors gracefully"""
        manager = clean_manager
        
        # Create server that raises exception
        error_server = Mock()
        error_server.send_messages = Mock(side_effect=Exception("Processing error"))
        
        initialize_queue(error_server)
        
        # Enqueue message
        put_messages(
            actor=sample_client,
            agent_id="agent-error",
            input_messages=sample_messages
        )
        
        # Wait for processing attempt
        time.sleep(1.5)
        
        # Worker should still be running despite error
        assert manager._worker._running
        
        # Cleanup
        manager.cleanup()


# ============================================================================
# Performance Tests
# ============================================================================

class TestQueuePerformance:
    """Performance tests for queue operations"""
    
    def test_enqueue_performance(self, clean_manager, sample_client, sample_messages):
        """Test enqueueing performance"""
        manager = clean_manager
        manager.initialize()
        
        start = time.time()
        
        # Enqueue 100 messages
        for i in range(100):
            put_messages(
                actor=sample_client,
                agent_id=f"agent-{i}",
                input_messages=sample_messages
            )
        
        elapsed = time.time() - start
        
        # Should be very fast (< 1 second for 100 messages)
        assert elapsed < 1.0
        
        print(f"\nEnqueued 100 messages in {elapsed:.3f}s")
        
        # Cleanup
        manager.cleanup()
    
    def test_concurrent_enqueue(self, sample_client, sample_messages, mock_server):
        """Test concurrent enqueueing from multiple threads"""
        # Initialize the global queue manager with mock server
        # This prevents messages from being discarded
        initialize_queue(server=mock_server)
        manager = get_manager()
        
        # Track processed messages
        processed_messages = []
        processed_lock = threading.Lock()
        
        def mock_send_messages(**kwargs):
            with processed_lock:
                processed_messages.append(kwargs['agent_id'])
            return None
        
        mock_server.send_messages = mock_send_messages
        
        def enqueue_messages(thread_id, count):
            for i in range(count):
                put_messages(
                    actor=sample_client,
                    agent_id=f"agent-{thread_id}-{i}",
                    input_messages=sample_messages
                )
        
        # Start multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=enqueue_messages, args=(i, 20))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Wait a bit for worker to process all messages
        time.sleep(1.0)
        
        # Verify all 100 messages were processed
        assert len(processed_messages) == 100  # 5 threads * 20 messages
        
        # Cleanup
        manager.cleanup()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

