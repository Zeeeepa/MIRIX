import React, { useState, useEffect, useCallback } from 'react';
import ReactFlow, {
  MiniMap,
  Controls,
  Background,
  BackgroundVariant,
  useNodesState,
  useEdgesState,
  addEdge,
  Node,
  Edge,
} from 'reactflow';
import 'reactflow/dist/style.css';
import './MemoryTreeVisualization.css';

interface MemoryItem {
  id?: string;
  title?: string;
  name?: string;
  filename?: string;
  summary?: string;
  details?: string;
  content?: string;
  timestamp?: string;
  tree_path?: string[];
  steps?: string[];
  tags?: string[];
  created_at?: string;
}

interface MemoryTreeVisualizationProps {
  memoryType: string;
  serverUrl?: string;
  getItemTitle?: (item: MemoryItem) => string;
  getItemDetails?: (item: MemoryItem) => { summary?: string; details?: string };
  refreshTrigger?: number;
}

interface NodeData {
  label: string;
  type: 'category' | 'memory-item';
  count?: number;
  items?: MemoryItem[];
  path?: string[];
  isExpandable?: boolean;
  childCount?: number;
  originalLabel?: string;
  memoryItem?: MemoryItem;
  parentPath?: string;
}

const MemoryTreeVisualization: React.FC<MemoryTreeVisualizationProps> = ({ 
  memoryType, 
  serverUrl = 'http://localhost:8000',
  getItemTitle = (item) => item.title || item.name || item.filename || item.summary || 'Memory Item',
  getItemDetails = (item) => ({ summary: item.summary, details: item.details }),
  refreshTrigger = 0
}) => {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedNode, setSelectedNode] = useState<Node<NodeData> | null>(null);
  const [expandedCategories, setExpandedCategories] = useState(new Set<string>());
  const [allNodes, setAllNodes] = useState<Node<NodeData>[]>([]);
  const [allEdges, setAllEdges] = useState<Edge[]>([]);
  const [nodes, setNodes, onNodesChange] = useNodesState<NodeData>([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);

  useEffect(() => {
    loadMemoryData();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [serverUrl, memoryType, refreshTrigger]);

  const loadMemoryData = async () => {
    try {
      setLoading(true);
      setError(null);
      
      const endpointMap: Record<string, string> = {
        'episodic': '/memory/episodic',
        'procedural': '/memory/procedural', 
        'resource': '/memory/resources',
        'semantic': '/memory/semantic'
      };
      
      const endpoint = endpointMap[memoryType];
      if (!endpoint) {
        throw new Error(`Unknown memory type: ${memoryType}`);
      }

      const response = await fetch(`${serverUrl}${endpoint}`);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      
      // Reset expanded categories when refreshing
      const newExpandedCategories = new Set<string>();
      setExpandedCategories(newExpandedCategories);
      
      // Build graph structure
      const { nodes: graphNodes, edges: graphEdges } = buildGraphFromItems(data);
      setAllNodes(graphNodes);
      setAllEdges(graphEdges);
      
      // Initially show only category nodes (collapsed state)
      updateVisibleNodes(graphNodes, graphEdges, new Set());
      
      setLoading(false);
    } catch (err) {
      setError(`Failed to load ${memoryType} memory: ${err instanceof Error ? err.message : 'Unknown error'}`);
      setLoading(false);
    }
  };

  const buildGraphFromItems = (items: MemoryItem[]) => {
    const nodes: Node<NodeData>[] = [];
    const edges: Edge[] = [];
    const nodeMap = new Map<string, Node<NodeData>>();
    
    // Build hierarchy from tree paths
    const pathGroups: Record<string, { path: string[], items: MemoryItem[] }> = {};
    
    items.forEach(item => {
      const treePath = item.tree_path && item.tree_path.length > 0 ? item.tree_path : ['uncategorized'];
      const pathKey = treePath.join('/');
      
      if (!pathGroups[pathKey]) {
        pathGroups[pathKey] = {
          path: treePath,
          items: []
        };
      }
      pathGroups[pathKey].items.push(item);
    });

    // Calculate layout dimensions
    const levelHeight = 120;
    const nodeWidth = 180;
    const nodeSpacing = 40;
    
    // Build tree structure with proper hierarchical positioning
    const levelNodes: Record<number, Node<NodeData>[]> = {};
    
    Object.values(pathGroups).forEach((group) => {
      let parentId: string | null = null;
      let currentPath = '';
      
      // Only create nodes for the path, excluding the final level (which will be memory items)
      group.path.slice(0, -1).forEach((segment, levelIndex) => {
        const nodePath = currentPath ? `${currentPath}/${segment}` : segment;
        
        if (!nodeMap.has(nodePath)) {
          // Initialize level if not exists
          if (!levelNodes[levelIndex]) {
            levelNodes[levelIndex] = [];
          }
          
          const node: Node<NodeData> = {
            id: nodePath,
            type: 'default',
            data: { 
              label: segment,
              type: 'category',
              count: 0,
              items: [],
              path: group.path.slice(0, levelIndex + 1),
              isExpandable: false,
            },
            position: { x: 0, y: 0 },
            className: 'category-node',
          };
          
          nodes.push(node);
          nodeMap.set(nodePath, node);
          levelNodes[levelIndex].push(node);
          
          // Create edge from parent to this node
          if (parentId) {
            edges.push({
              id: `${parentId}-${nodePath}`,
              source: parentId,
              target: nodePath,
              type: 'smoothstep',
              animated: false,
            });
          }
        }
        
        parentId = nodePath;
        currentPath = nodePath;
      });
    });

    // Group memory items by their parent for horizontal positioning
    const memoryItemsByParent: Record<string, { item: MemoryItem, group: { path: string[], items: MemoryItem[] } }[]> = {};
    Object.values(pathGroups).forEach((group) => {
      if (group.items.length > 0) {
        const parentPath = group.path.slice(0, -1).join('/');
        if (!memoryItemsByParent[parentPath]) {
          memoryItemsByParent[parentPath] = [];
        }
        
        group.items.forEach((item) => {
          memoryItemsByParent[parentPath].push({
            item,
            group
          });
        });
      }
    });

    // Mark category nodes as expandable if they have memory items as children
    Object.keys(memoryItemsByParent).forEach((parentPath) => {
      const parentNode = nodeMap.get(parentPath);
      if (parentNode) {
        parentNode.data.isExpandable = true;
        parentNode.data.childCount = memoryItemsByParent[parentPath].length;
        parentNode.data.originalLabel = parentNode.data.label;
        // Update label to show count and expand/collapse indicator
        parentNode.data.label = `${parentNode.data.originalLabel} (${parentNode.data.childCount}) +`;
      }
    });

    // Now add individual memory items as bottom-level nodes
    const memoryItemLevel = Math.max(...Object.keys(levelNodes).map(Number)) + 1;
    levelNodes[memoryItemLevel] = [];
    
    Object.entries(memoryItemsByParent).forEach(([parentPath, items]) => {
      items.forEach((itemData, itemIndex) => {
        const itemId = `${itemData.group.path.join('/')}/item-${itemIndex}`;
        const itemNode: Node<NodeData> = {
          id: itemId,
          type: 'output',
          data: {
            label: getItemTitle(itemData.item),
            type: 'memory-item',
            memoryItem: itemData.item,
            path: itemData.group.path,
            parentPath: parentPath
          },
          position: { x: 0, y: 0 },
          className: 'memory-item-node',
        };
        
        nodes.push(itemNode);
        levelNodes[memoryItemLevel].push(itemNode);
        
        // Connect to parent category
        edges.push({
          id: `${parentPath}-${itemId}`,
          source: parentPath,
          target: itemId,
          type: 'smoothstep',
          animated: false,
        });
      });
    });

    // Calculate positions for hierarchical layout
    const maxLevel = Math.max(...Object.keys(levelNodes).map(Number));
    const startY = 80;
    
    // Position nodes level by level
    for (let levelIndex = 0; levelIndex <= maxLevel; levelIndex++) {
      const nodesAtLevel = levelNodes[levelIndex] || [];
      
      if (levelIndex === 0) {
        // Top level - position evenly across the screen
        const totalWidth = nodesAtLevel.length * (nodeWidth + nodeSpacing) - nodeSpacing;
        const startX = Math.max(100, (1200 - totalWidth) / 2);
        
        nodesAtLevel.forEach((node, index) => {
          node.position = {
            x: startX + index * (nodeWidth + nodeSpacing),
            y: startY + levelIndex * levelHeight
          };
        });
      } else if (levelIndex === maxLevel) {
        // Memory items - position under their direct parents
        const itemsByParent: Record<string, Node<NodeData>[]> = {};
        nodesAtLevel.forEach(node => {
          const parentPath = node.data.parentPath;
          if (parentPath) {
            if (!itemsByParent[parentPath]) {
              itemsByParent[parentPath] = [];
            }
            itemsByParent[parentPath].push(node);
          }
        });
        
        Object.entries(itemsByParent).forEach(([parentPath, items]) => {
          const parentNode = nodes.find(n => n.id === parentPath);
          if (parentNode) {
            const parentX = parentNode.position.x;
            const itemsWidth = items.length * (nodeWidth + nodeSpacing) - nodeSpacing;
            const startX = parentX - itemsWidth / 2 + nodeWidth / 2;
            
            items.forEach((item, index) => {
              item.position = {
                x: startX + index * (nodeWidth + nodeSpacing),
                y: startY + levelIndex * levelHeight
              };
            });
          }
        });
      } else {
        // Category nodes - distribute evenly
        const totalWidth = nodesAtLevel.length * (nodeWidth + nodeSpacing) - nodeSpacing;
        const startX = Math.max(100, (1200 - totalWidth) / 2);
        
        nodesAtLevel.forEach((node, index) => {
          node.position = {
            x: startX + index * (nodeWidth + nodeSpacing),
            y: startY + levelIndex * levelHeight
          };
        });
      }
    }

    return { nodes, edges };
  };

  const updateVisibleNodes = (allNodes: Node<NodeData>[], allEdges: Edge[], expandedCategories: Set<string>) => {
    // Filter nodes: show all category nodes and only memory items whose parents are expanded
    const visibleNodes = allNodes.filter(node => {
      if (node.data.type === 'category') {
        // Update the label based on expansion state
        if (node.data.isExpandable) {
          const isExpanded = expandedCategories.has(node.id);
          node.data.label = `${node.data.originalLabel} (${node.data.childCount}) ${isExpanded ? '−' : '+'}`;
        }
        return true; // Always show category nodes
      } else if (node.data.type === 'memory-item') {
        // Only show memory items if their parent is expanded
        return node.data.parentPath ? expandedCategories.has(node.data.parentPath) : false;
      }
      return true;
    });

    // Filter edges: only show edges between visible nodes
    const visibleNodeIds = new Set(visibleNodes.map(n => n.id));
    const visibleEdges = allEdges.filter(edge => 
      visibleNodeIds.has(edge.source) && visibleNodeIds.has(edge.target)
    );

    setNodes(visibleNodes);
    setEdges(visibleEdges);
  };

  const onConnect = useCallback(
    (params: any) => setEdges((eds) => addEdge(params, eds)),
    [setEdges],
  );

  const onNodeClick = useCallback((event: React.MouseEvent, node: Node<NodeData>) => {
    if (node.data.type === 'category' && node.data.isExpandable) {
      // Toggle expansion for category nodes
      setExpandedCategories(prev => {
        const newExpanded = new Set(prev);
        if (newExpanded.has(node.id)) {
          newExpanded.delete(node.id);
        } else {
          newExpanded.add(node.id);
        }
        // Update visible nodes with new expansion state
        updateVisibleNodes(allNodes, allEdges, newExpanded);
        return newExpanded;
      });
    } else if (node.data.type === 'memory-item') {
      // Set selected node for memory items
      setSelectedNode(node);
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [allNodes, allEdges]);

  const renderNodeDetails = () => {
    if (!selectedNode || selectedNode.data.type !== 'memory-item') {
      return null;
    }

    const { data } = selectedNode;
    const memoryItem = data.memoryItem;
    const itemDetails = memoryItem ? getItemDetails(memoryItem) : null;
    
    return (
      <div className="node-details-panel">
        <h3>{data.label}</h3>
        
        {memoryItem && (
          <div className="memory-item-details">
            {memoryType === 'episodic' && memoryItem.timestamp && (
              <div className="detail-section">
                <h4>Timestamp</h4>
                <p>{new Date(memoryItem.timestamp).toLocaleString()}</p>
              </div>
            )}
            
            {memoryType === 'procedural' && memoryItem.steps && (
              <div className="detail-section">
                <h4>Steps</h4>
                <ol>
                  {memoryItem.steps.map((step, index) => (
                    <li key={index}>{step}</li>
                  ))}
                </ol>
              </div>
            )}
            
            {memoryType === 'semantic' && memoryItem.tags && memoryItem.tags.length > 0 && (
              <div className="detail-section">
                <h4>Tags</h4>
                <p>{memoryItem.tags.join(', ')}</p>
              </div>
            )}
            
            {itemDetails?.summary && (
              <div className="detail-section">
                <h4>Summary</h4>
                <p>{itemDetails.summary}</p>
              </div>
            )}
            
            {itemDetails?.details && (
              <div className="detail-section">
                <h4>Details</h4>
                <p>{itemDetails.details}</p>
              </div>
            )}

            <button 
              className="sidebar-close-button"
              onClick={() => setSelectedNode(null)}
              title="Close sidebar"
            >
              Close
            </button>
          </div>
        )}
      </div>
    );
  };

  if (loading) {
    return <div className="memory-tree-loading">Loading {memoryType} memory tree...</div>;
  }

  if (error) {
    return <div className="memory-tree-error">Error: {error}</div>;
  }

  return (
    <div className="memory-tree-visualization">
      <div className="graph-container">
        <ReactFlow
          nodes={nodes}
          edges={edges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onConnect={onConnect}
          onNodeClick={onNodeClick}
          fitView
          minZoom={0.1}
          maxZoom={2}
        >
          <Controls />
          <MiniMap 
            nodeColor={(node: any) => {
              switch (node.data?.type) {
                case 'category': return '#4ecdc4';
                case 'memory-item': return '#45b7d1';
                default: return '#96ceb4';
              }
            }}
          />
          <Background variant={BackgroundVariant.Dots} gap={12} size={1} />
        </ReactFlow>
      </div>
      {selectedNode && selectedNode.data.type === 'memory-item' && (
        <div className="details-sidebar">
          {renderNodeDetails()}
        </div>
      )}
    </div>
  );
};

export default MemoryTreeVisualization; 