import React, { useState } from 'react';
import './App.css';
import UploadContentSection from './components/UploadContentSection';
import MemoryVisualizationPanel from './components/MemoryVisualizationPanel';
import ChatInterface from './components/ChatInterface';

interface Settings {
  serverUrl: string;
}

function App() {
  const [settings] = useState<Settings>({
    serverUrl: 'http://localhost:8000'
  });

  const [refreshTrigger, setRefreshTrigger] = useState(0);
  const [showMemoryPanel, setShowMemoryPanel] = useState(false);

  const handleContentUploaded = () => {
    // Trigger memory visualization refresh
    setRefreshTrigger(prev => prev + 1);
  };

  const toggleMemoryPanel = () => {
    setShowMemoryPanel(prev => !prev);
  };

  return (
    <div className="App">
      <header className="app-header">
        <div className="header-content">
          <div className="header-text">
            <h1>Memory Interface</h1>
            <p>Upload content, visualize memory structure, and chat with your agent</p>
          </div>
          <button 
            onClick={toggleMemoryPanel}
            className="memory-toggle-btn"
            title={showMemoryPanel ? "Hide Memory Structure" : "Show Memory Structure"}
          >
            {showMemoryPanel ? "🧠 Hide Memory" : "🧠 Show Memory"}
          </button>
        </div>
      </header>
      
      <div className={`app-layout ${showMemoryPanel ? 'with-memory' : 'without-memory'}`}>
        {/* Left side: Upload section */}
        <div className="upload-section">
          <UploadContentSection 
            settings={settings} 
            onContentUploaded={handleContentUploaded}
          />
        </div>
        
        {/* Center: Memory visualization - conditionally rendered */}
        {showMemoryPanel && (
          <div className="memory-section">
            <MemoryVisualizationPanel 
              settings={settings}
              refreshTrigger={refreshTrigger}
            />
          </div>
        )}
        
        {/* Right side: Chat interface */}
        <div className="chat-section">
          <ChatInterface settings={settings} />
        </div>
      </div>
    </div>
  );
}

export default App;
