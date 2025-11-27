import React, { useEffect, useState } from 'react';
import { useWorkspace } from '@/contexts/WorkspaceContext';
import apiClient from '@/api/client';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Search, Brain } from 'lucide-react';

interface MemoryResult {
  memory_type: string;
  id: string;
  summary?: string;
  details?: string;
  content?: string;
  caption?: string;
  timestamp?: string;
  created_at?: string;
}

export const Memories: React.FC = () => {
  const { selectedUser } = useWorkspace();
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<MemoryResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [hasSearched, setHasSearched] = useState(false);

  const handleSearch = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (!selectedUser || !query.trim()) return;
    
    setLoading(true);
    setHasSearched(true);
    try {
      const response = await apiClient.get('/memory/search', {
        params: {
          user_id: selectedUser.id,
          query: query,
          memory_type: 'all',
          limit: 20
        }
      });
      setResults(response.data.results);
    } catch (error) {
      console.error('Error searching memories:', error);
    } finally {
      setLoading(false);
    }
  };

  // Reset search when user changes
  useEffect(() => {
    setResults([]);
    setHasSearched(false);
    setQuery('');
  }, [selectedUser?.id]);

  if (!selectedUser) {
    return (
      <div className="text-center py-12">
        <h2 className="text-2xl font-bold text-muted-foreground">Please select a user to view memories</h2>
      </div>
    );
  }

  return (
    <div className="space-y-8">
      <div>
        <h2 className="text-3xl font-bold tracking-tight">Memories: {selectedUser.name}</h2>
        <p className="text-muted-foreground">
          Search and explore what your agent remembers about {selectedUser.name}.
        </p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Search Memories</CardTitle>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSearch} className="flex gap-4">
            <Input
              placeholder="Search (e.g. 'meeting notes', 'project details')"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              className="flex-1"
            />
            <Button type="submit" disabled={loading || !selectedUser}>
              <Search className="mr-2 h-4 w-4" /> Search
            </Button>
          </form>
        </CardContent>
      </Card>

      <div className="space-y-4">
        {loading ? (
          <div className="text-center py-8">Searching...</div>
        ) : hasSearched && results.length === 0 ? (
          <div className="text-center py-8 text-muted-foreground">No memories found matching your query.</div>
        ) : (
          results.map((memory) => (
            <Card key={memory.id} className="overflow-hidden">
              <div className="border-l-4 border-primary pl-4 py-4 pr-4">
                <div className="flex items-center gap-2 mb-2">
                  <span className="text-xs font-bold uppercase tracking-wider text-muted-foreground bg-muted px-2 py-0.5 rounded">
                    {memory.memory_type}
                  </span>
                  {memory.timestamp && (
                    <span className="text-xs text-muted-foreground">
                      {new Date(memory.timestamp).toLocaleString()}
                    </span>
                  )}
                </div>
                <h4 className="font-medium text-lg mb-1">
                  {memory.summary || memory.caption || 'No summary'}
                </h4>
                <p className="text-sm text-muted-foreground line-clamp-3">
                  {memory.details || memory.content || 'No additional details.'}
                </p>
              </div>
            </Card>
          ))
        )}
        
        {!hasSearched && (
          <div className="text-center py-12 border-2 border-dashed rounded-lg">
            <Brain className="mx-auto h-12 w-12 text-muted-foreground/50 mb-4" />
            <h3 className="text-lg font-medium">Ready to Search</h3>
            <p className="text-muted-foreground">Enter a query above to find specific memories for {selectedUser.name}.</p>
          </div>
        )}
      </div>
    </div>
  );
};

