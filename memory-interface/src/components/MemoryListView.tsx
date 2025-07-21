import React, { useState, useEffect } from 'react';
import './MemoryListView.css';

interface MemoryItem {
  id?: string;
  title?: string;
  name?: string;
  filename?: string;
  summary?: string;
  details?: string;
  content?: string;
  timestamp?: string;
  aspect?: string;
  category?: string;
  understanding?: string;
  caption?: string;
  entry_type?: string;
  source?: string;
  sensitivity?: string;
  steps?: string[];
  tags?: string[];
  created_at?: string;
  last_updated?: string;
  last_accessed?: string;
  size?: string;
  type?: string;
  proficiency?: string;
  difficulty?: string;
  success_rate?: string;
  time_to_complete?: string;
  last_practiced?: string;
  prerequisites?: string[];
}

interface MemoryListViewProps {
  memoryType: string;
  serverUrl: string;
  refreshTrigger: number;
}

const MemoryListView: React.FC<MemoryListViewProps> = ({ 
  memoryType, 
  serverUrl, 
  refreshTrigger 
}) => {
  const [memoryData, setMemoryData] = useState<MemoryItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [expandedItems, setExpandedItems] = useState(new Set<string>());

  useEffect(() => {
    fetchMemoryData();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [memoryType, serverUrl, refreshTrigger]);

  const fetchMemoryData = async () => {
    try {
      setLoading(true);
      setError(null);

      const endpointMap: Record<string, string> = {
        'past-events': '/memory/episodic',
        'semantic': '/memory/semantic',
        'procedural': '/memory/procedural',
        'docs-files': '/memory/resources',
        'core-understanding': '/memory/core',
        'credentials': '/memory/credentials'
      };

      const endpoint = endpointMap[memoryType];
      if (!endpoint) {
        throw new Error(`Unknown memory type: ${memoryType}`);
      }

      const response = await fetch(`${serverUrl}${endpoint}`);
      if (!response.ok) {
        throw new Error(`Failed to fetch ${memoryType}: ${response.statusText}`);
      }

      const data = await response.json();
      setMemoryData(data);
    } catch (err) {
      console.error(`Error fetching ${memoryType}:`, err);
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  };

  const filterMemories = (memories: MemoryItem[], query: string) => {
    if (!query.trim()) {
      return memories;
    }

    const searchTerm = query.toLowerCase();
    
    return memories.filter(item => {
      const searchableText = [
        item.content,
        item.summary,
        item.title,
        item.name,
        item.filename,
        item.aspect,
        item.category,
        item.understanding,
        item.caption,
        item.entry_type,
        item.source,
        ...(item.tags || []),
        ...(item.steps || [])
      ]
        .filter(Boolean)
        .join(' ')
        .toLowerCase();

      return searchableText.includes(searchTerm);
    });
  };

  const highlightText = (text: string | undefined, query: string) => {
    if (!text || !query.trim()) {
      return text;
    }

    const searchTerm = query.trim();
    const regex = new RegExp(`(${searchTerm.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')})`, 'gi');
    const parts = text.split(regex);

    return parts.map((part, index) => 
      regex.test(part) ? (
        <span key={index} className="search-highlight">{part}</span>
      ) : part
    );
  };

  const toggleExpanded = (itemId: string) => {
    setExpandedItems(prev => {
      const newSet = new Set(prev);
      if (newSet.has(itemId)) {
        newSet.delete(itemId);
      } else {
        newSet.add(itemId);
      }
      return newSet;
    });
  };

  const renderMemoryItem = (item: MemoryItem, index: number) => {
    const itemId = `${memoryType}-${index}`;
    const isExpanded = expandedItems.has(itemId);

    switch (memoryType) {
      case 'past-events':
        return (
          <div key={index} className="memory-item semantic-memory">
            <div className="memory-title">
              {item.timestamp ? new Date(item.timestamp).toLocaleString() : 'Unknown time'}
            </div>
            {item.summary && <div className="memory-summary">{highlightText(item.summary, searchQuery)}</div>}
            {item.details && (
              <div className="memory-details-section">
                <button 
                  className="expand-toggle-button"
                  onClick={() => toggleExpanded(itemId)}
                  title={isExpanded ? "Collapse details" : "Expand details"}
                >
                  {isExpanded ? "▼ Hide Details" : "▶ Show Details"}
                </button>
                {isExpanded && (
                  <div className="memory-details">{highlightText(item.details, searchQuery)}</div>
                )}
              </div>
            )}
          </div>
        );

      case 'semantic':
        return (
          <div key={index} className="memory-item semantic-memory">
            <div className="memory-title">{highlightText(item.title || item.name, searchQuery)}</div>
            {item.summary && <div className="memory-summary">{highlightText(item.summary, searchQuery)}</div>}
            {item.details && (
              <div className="memory-details-section">
                <button 
                  className="expand-toggle-button"
                  onClick={() => toggleExpanded(itemId)}
                  title={isExpanded ? "Collapse details" : "Expand details"}
                >
                  {isExpanded ? "▼ Hide Details" : "▶ Show Details"}
                </button>
                {isExpanded && (
                  <div className="memory-details">{highlightText(item.details, searchQuery)}</div>
                )}
              </div>
            )}
            {item.last_updated && <div className="memory-updated">Updated: {new Date(item.last_updated).toLocaleString()}</div>}
            {item.tags && (
              <div className="memory-tags">
                {item.tags.map((tag, i) => (
                  <span key={i} className="memory-tag">{highlightText(tag, searchQuery)}</span>
                ))}
              </div>
            )}
          </div>
        );

      case 'procedural':
        return (
          <div key={index} className="memory-item procedural-memory">
            <div className="memory-title">{highlightText(item.summary, searchQuery)}</div>
            <div className="memory-content">
              {item.steps && item.steps.length > 0 ? (
                <div className="memory-steps">
                  <strong>🎯 Step-by-Step Guide:</strong>
                  <ol>
                    {item.steps.map((step, i) => (
                      <li key={i}>{highlightText(step, searchQuery)}</li>
                    ))}
                  </ol>
                </div>
              ) : (
                <div>{highlightText(item.content || 'No steps available', searchQuery)}</div>
              )}
            </div>
            {item.proficiency && <div className="memory-proficiency">Proficiency: {highlightText(item.proficiency, searchQuery)}</div>}
            {item.difficulty && <div className="memory-difficulty">Difficulty: {highlightText(item.difficulty, searchQuery)}</div>}
            {item.success_rate && <div className="memory-success-rate">Success Rate: {highlightText(item.success_rate, searchQuery)}</div>}
            {item.time_to_complete && <div className="memory-time">Time to Complete: {highlightText(item.time_to_complete, searchQuery)}</div>}
            {item.last_practiced && <div className="memory-practiced">Last Practiced: {new Date(item.last_practiced).toLocaleString()}</div>}
            {item.prerequisites && item.prerequisites.length > 0 && (
              <div className="memory-prerequisites">
                <strong>Prerequisites:</strong> {item.prerequisites.map(prereq => highlightText(prereq, searchQuery)).join(', ')}
              </div>
            )}
            {item.last_updated && <div className="memory-updated">Updated: {new Date(item.last_updated).toLocaleString()}</div>}
            {item.tags && (
              <div className="memory-tags">
                {item.tags.map((tag, i) => (
                  <span key={i} className="memory-tag">{highlightText(tag, searchQuery)}</span>
                ))}
              </div>
            )}
          </div>
        );

      case 'docs-files':
        return (
          <div key={index} className="memory-item resource-memory">
            <div className="memory-filename">{highlightText(item.filename || item.name, searchQuery)}</div>
            <div className="memory-file-type">{highlightText(item.type || 'Unknown', searchQuery)}</div>
            <div className="memory-summary">{highlightText(item.summary || item.content, searchQuery)}</div>
            {item.last_accessed && (
              <div className="memory-accessed">Last accessed: {new Date(item.last_accessed).toLocaleString()}</div>
            )}
            {item.size && <div className="memory-size">Size: {item.size}</div>}
          </div>
        );

      case 'core-understanding':
        return (
          <div key={index} className="memory-item core-memory">
            <div className="memory-aspect-header">
              <div className="memory-aspect">
                {highlightText(item.aspect || item.category, searchQuery)}
              </div>
            </div>
            <div className="memory-understanding">
              {highlightText(item.understanding || item.content, searchQuery)}
            </div>
            {item.last_updated && (
              <div className="memory-updated">Updated: {new Date(item.last_updated).toLocaleString()}</div>
            )}
          </div>
        );

      case 'credentials':
        return (
          <div key={index} className="memory-item credential-memory">
            <div className="memory-credential-name">{highlightText(item.caption, searchQuery)}</div>
            <div className="memory-credential-type">{highlightText(item.entry_type || 'Credential', searchQuery)}</div>
            <div className="memory-credential-content">
              {item.content || 'Content masked for security'}
            </div>
            {item.source && (
              <div className="memory-credential-source">Source: {highlightText(item.source, searchQuery)}</div>
            )}
            {item.sensitivity && (
              <div className="memory-credential-sensitivity">
                <span className={`sensitivity-badge sensitivity-${item.sensitivity}`}>
                  {item.sensitivity.charAt(0).toUpperCase() + item.sensitivity.slice(1)} Sensitivity
                </span>
              </div>
            )}
          </div>
        );

      default:
        return (
          <div key={index} className="memory-item">
            <div className="memory-content">{JSON.stringify(item, null, 2)}</div>
          </div>
        );
    }
  };

  const getMemoryTypeLabel = (type: string) => {
    switch (type) {
      case 'past-events': return 'Episodic';
      case 'semantic': return 'Semantic';
      case 'procedural': return 'Procedural';
      case 'docs-files': return 'Resource';
      case 'core-understanding': return 'Core';
      case 'credentials': return 'Credentials';
      default: return 'Memory';
    }
  };

  const filteredData = filterMemories(memoryData, searchQuery);

  if (loading) {
    return (
      <div className="memory-list-loading">
        <div className="loading-spinner"></div>
        <p>Loading {getMemoryTypeLabel(memoryType).toLowerCase()} memory...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="memory-list-error">
        <p>Error loading memory: {error}</p>
        <button onClick={fetchMemoryData} className="retry-button">
          Retry
        </button>
      </div>
    );
  }

  return (
    <div className="memory-list-view">
      <div className="memory-search-section">
        <div className="search-input-container">
          <span className="search-icon">🔍</span>
          <input
            type="text"
            placeholder={`Search ${getMemoryTypeLabel(memoryType).toLowerCase()}...`}
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="search-input"
          />
          {searchQuery && (
            <button
              onClick={() => setSearchQuery('')}
              className="clear-search-button"
              title="Clear search"
            >
              ✕
            </button>
          )}
        </div>
        
        <button 
          onClick={fetchMemoryData} 
          className="refresh-button"
          disabled={loading}
        >
          🔄 Refresh
        </button>
      </div>

      <div className="memory-items-container">
        {memoryData.length === 0 ? (
          <div className="memory-empty">
            <p>No {getMemoryTypeLabel(memoryType).toLowerCase()} found.</p>
          </div>
        ) : filteredData.length === 0 && searchQuery.trim() ? (
          <div className="memory-empty">
            <p>No {getMemoryTypeLabel(memoryType).toLowerCase()} found matching "{searchQuery}".</p>
            <p>Try a different search term or clear the search to see all memories.</p>
          </div>
        ) : (
          filteredData.map((item, index) => renderMemoryItem(item, index))
        )}
      </div>
    </div>
  );
};

export default MemoryListView; 