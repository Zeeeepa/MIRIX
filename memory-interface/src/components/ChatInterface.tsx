import React, { useState, useRef, useEffect } from 'react';
import './ChatInterface.css';

interface Settings {
  serverUrl: string;
}

interface ChatInterfaceProps {
  settings: Settings;
}

interface ContentPart {
  text?: string;
  image_data?: string;  // Base64 data URI
  image_path?: string;  // Original file path
  file_path?: string;   // File path
  file_size?: number;   // File size in bytes
  file_mime_type?: string; // MIME type
}

interface Message {
  id: string;
  type: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  contentParts?: ContentPart[];  // For structured responses with images/files
}

// Component to render structured content with images and files
const StructuredContent: React.FC<{ contentParts: ContentPart[] }> = ({ contentParts }) => {
  return (
    <div className="structured-content">
      {contentParts.map((part, index) => (
        <div key={index} className="content-part">
          {part.text && (
            <div className="text-content">
              {part.text}
            </div>
          )}
          
          {part.image_data && (
            <div className="image-content">
              <img 
                src={part.image_data} 
                alt="Attached image"
                style={{
                  maxWidth: '100%',
                  maxHeight: '400px',
                  borderRadius: '8px',
                  marginTop: '8px',
                  marginBottom: '8px'
                }}
                onError={(e) => {
                  console.error('Failed to load image:', part.image_path);
                  (e.target as HTMLImageElement).style.display = 'none';
                }}
              />
            </div>
          )}
          
          {part.file_path && !part.image_data && (
            <div className="file-content">
              <div className="file-attachment">
                <span className="file-icon">📎</span>
                <div className="file-info">
                  <div className="file-name">{part.file_path.split('/').pop()}</div>
                  {part.file_size && (
                    <div className="file-size">{(part.file_size / 1024).toFixed(1)} KB</div>
                  )}
                  {part.file_mime_type && (
                    <div className="file-type">{part.file_mime_type}</div>
                  )}
                </div>
              </div>
            </div>
          )}
        </div>
      ))}
    </div>
  );
};

const ChatInterface: React.FC<ChatInterfaceProps> = ({ settings }) => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const addMessage = (type: 'user' | 'assistant', content: string, contentParts?: ContentPart[]) => {
    const newMessage: Message = {
      id: Date.now().toString(),
      type,
      content,
      timestamp: new Date(),
      contentParts
    };
    setMessages(prev => [...prev, newMessage]);
  };

  const sendMessage = async () => {
    if (!inputValue.trim() || isLoading) return;

    const userMessage = inputValue.trim();
    setInputValue('');
    setError(null);
    
    // Add user message immediately
    addMessage('user', userMessage);
    setIsLoading(true);

    try {
      const response = await fetch(`${settings.serverUrl}/send_message`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: userMessage,
          memorizing: false  // This is for Q&A, not memorization
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      
      if (result.status === 'success') {
        // Handle structured responses with images/files
        if (result.content_parts && result.content_parts.length > 0) {
          addMessage('assistant', result.response || 'No response received', result.content_parts);
        } else {
          addMessage('assistant', result.response || 'No response received');
        }
      } else if (result.status === 'missing_api_keys') {
        addMessage('assistant', `Missing API keys: ${result.missing_keys?.join(', ') || 'Unknown keys required'}`);
      } else {
        addMessage('assistant', `Error: ${result.response || 'Unknown error occurred'}`);
      }

    } catch (err) {
      console.error('Error sending message:', err);
      const errorMessage = err instanceof Error ? err.message : 'Failed to send message';
      setError(errorMessage);
      addMessage('assistant', `Error: ${errorMessage}`);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const clearChat = () => {
    setMessages([]);
    setError(null);
  };

  const formatTimestamp = (timestamp: Date) => {
    return timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  return (
    <div className="chat-interface">
      <div className="chat-header">
        <div className="chat-title">
          <h2>Chat with Agent</h2>
          <p>Ask questions about your memories</p>
        </div>
        <button 
          onClick={clearChat}
          className="clear-chat-btn"
          disabled={messages.length === 0}
        >
          🗑️ Clear
        </button>
      </div>

      <div className="chat-messages">
        {messages.length === 0 ? (
          <div className="chat-empty">
            <div className="empty-state">
              <span className="empty-icon">💬</span>
              <p>Start a conversation!</p>
              <small>Ask questions about your memories, get insights, or have a general chat.</small>
            </div>
          </div>
        ) : (
          messages.map((message) => (
            <div
              key={message.id}
              className={`message ${message.type === 'user' ? 'user-message' : 'assistant-message'}`}
            >
              <div className="message-content">
                {message.contentParts && message.contentParts.length > 0 ? (
                  <StructuredContent contentParts={message.contentParts} />
                ) : (
                  <div className="message-text">{message.content}</div>
                )}
                <div className="message-timestamp">
                  {formatTimestamp(message.timestamp)}
                </div>
              </div>
            </div>
          ))
        )}
        
        {isLoading && (
          <div className="message assistant-message">
            <div className="message-content">
              <div className="message-text typing-indicator">
                <div className="typing-dots">
                  <span></span>
                  <span></span>
                  <span></span>
                </div>
                Agent is thinking...
              </div>
            </div>
          </div>
        )}
        
        <div ref={messagesEndRef} />
      </div>

      <div className="chat-input-section">
        {error && (
          <div className="chat-error">
            <span className="error-icon">⚠️</span>
            {error}
          </div>
        )}
        
        <div className="chat-input-container">
          <textarea
            ref={inputRef}
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Ask a question about your memories..."
            className="chat-input"
            rows={3}
            disabled={isLoading}
          />
          <button
            onClick={sendMessage}
            disabled={!inputValue.trim() || isLoading}
            className="send-button"
          >
            {isLoading ? '⏳' : '➤'}
          </button>
        </div>
        
        <div className="input-hint">
          Press Enter to send, Shift+Enter for new line
        </div>
      </div>
    </div>
  );
};

export default ChatInterface; 