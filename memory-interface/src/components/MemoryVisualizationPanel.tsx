import React, { useState } from 'react';
import MemoryTreeVisualization from './MemoryTreeVisualization';
import MemoryListView from './MemoryListView';
import './MemoryVisualizationPanel.css';

interface Settings {
  serverUrl: string;
}

interface MemoryVisualizationPanelProps {
  settings: Settings;
  refreshTrigger: number;
}

const MemoryVisualizationPanel: React.FC<MemoryVisualizationPanelProps> = ({ 
  settings,
  refreshTrigger
}) => {
  const [activeSubTab, setActiveSubTab] = useState('past-events');
  const [viewModes, setViewModes] = useState<Record<string, 'list' | 'tree'>>({
    'past-events': 'list',
    'semantic': 'list',
    'procedural': 'list',
    'docs-files': 'list',
    'core-understanding': 'list',
    'credentials': 'list'
  });

  const memoryTypes = [
    { id: 'past-events', label: 'Episodic', icon: '📅', description: 'Past Events' },
    { id: 'semantic', label: 'Semantic', icon: '🧠', description: 'Knowledge & Concepts' },
    { id: 'procedural', label: 'Procedural', icon: '🛠️', description: 'Skills & Procedures' },
    { id: 'docs-files', label: 'Resources', icon: '📁', description: 'Documents & Files' },
    { id: 'core-understanding', label: 'Core Memory', icon: '💡', description: 'Core Understanding' },
    { id: 'credentials', label: 'Knowledge Vault', icon: '🔐', description: 'Secure Information' },
  ];

  const getCurrentViewMode = () => viewModes[activeSubTab] || 'list';

  const setCurrentViewMode = (mode: 'list' | 'tree') => {
    setViewModes(prev => ({
      ...prev,
      [activeSubTab]: mode
    }));
  };

  const getItemTitle = (item: any) => {
    switch (activeSubTab) {
      case 'past-events':
        return item.summary || 'Episodic Event';
      case 'semantic':
        return item.title || item.name || item.summary || 'Semantic Item';
      case 'procedural': 
        return item.summary || item.title || 'Procedure';
      case 'docs-files':
        return item.filename || item.name || 'Resource';
      case 'core-understanding':
        return item.aspect || item.category || 'Core Memory';
      case 'credentials':
        return item.caption || item.name || 'Credential';
      default:
        return item.title || item.name || 'Memory Item';
    }
  };

  const getItemDetails = (item: any) => {
    return {
      summary: item.summary,
      details: item.details || item.content || item.understanding
    };
  };

  // Memory types that support tree view
  const supportsTreeView = ['past-events', 'semantic', 'procedural', 'docs-files'].includes(activeSubTab);

  return (
    <div className="memory-visualization-panel">
      <div className="memory-panel-header">
        <h2>Memory Structure</h2>
        <p>Explore your agent's knowledge as an interactive visualization</p>
      </div>

      <div className="memory-header">
        <div className="memory-subtabs">
          <div className="memory-subtabs-left">
            {memoryTypes.map(subTab => (
              <button
                key={subTab.id}
                className={`memory-subtab ${activeSubTab === subTab.id ? 'active' : ''}`}
                onClick={() => setActiveSubTab(subTab.id)}
                title={subTab.description}
              >
                <span className="subtab-icon">{subTab.icon}</span>
                <span className="subtab-label">{subTab.label}</span>
              </button>
            ))}
          </div>
        </div>
      </div>

      <div className="memory-content">
        <div className="memory-search-and-actions">
          {supportsTreeView && (
            <div className="view-mode-toggle">
              <button
                onClick={() => setCurrentViewMode('list')}
                className={`view-mode-button ${getCurrentViewMode() === 'list' ? 'active' : ''}`}
                title="List view"
              >
                📋 List
              </button>
              <button
                onClick={() => setCurrentViewMode('tree')}
                className={`view-mode-button ${getCurrentViewMode() === 'tree' ? 'active' : ''}`}
                title="Tree view"
              >
                🌳 Tree
              </button>
            </div>
          )}
        </div>

        <div className="memory-visualization-container">
          {getCurrentViewMode() === 'tree' && supportsTreeView ? (
            <MemoryTreeVisualization 
              memoryType={activeSubTab === 'past-events' ? 'episodic' : 
                         activeSubTab === 'docs-files' ? 'resource' : activeSubTab}
              serverUrl={settings.serverUrl}
              getItemTitle={getItemTitle}
              getItemDetails={getItemDetails}
              refreshTrigger={refreshTrigger}
            />
          ) : (
            <MemoryListView
              memoryType={activeSubTab}
              serverUrl={settings.serverUrl}
              refreshTrigger={refreshTrigger}
            />
          )}
        </div>
      </div>
    </div>
  );
};

export default MemoryVisualizationPanel; 