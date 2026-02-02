'use client';

import { useState } from 'react';
import { Search, Filter, Brain, Sparkles } from 'lucide-react';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { cn } from '@/lib/utils';

type SearchMode = 'filter' | 'semantic';

interface SearchBarProps {
  onSearch: (query: string) => void;
  onSemanticSearch?: (query: string) => void;
  isLoading?: boolean;
  placeholder?: string;
  showModeToggle?: boolean;
  defaultMode?: SearchMode;
}

const FILTER_EXAMPLES = [
  'high volatility',
  'bullish trend',
  'market crash',
  'low volatility',
  'rally',
  'downtrend',
  '2023-11',
];

const SEMANTIC_EXAMPLES = [
  'periods like current conditions',
  'before major corrections',
  'high fear low momentum',
  'after Fed rate hikes',
  'pre-earnings volatility',
];

export function SearchBar({
  onSearch,
  onSemanticSearch,
  isLoading = false,
  placeholder,
  showModeToggle = true,
  defaultMode = 'filter',
}: SearchBarProps) {
  const [query, setQuery] = useState('');
  const [mode, setMode] = useState<SearchMode>(defaultMode);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (mode === 'semantic' && onSemanticSearch) {
      onSemanticSearch(query);
    } else {
      onSearch(query);
    }
  };

  const handleExampleClick = (example: string) => {
    setQuery(example);
    if (mode === 'semantic' && onSemanticSearch) {
      onSemanticSearch(example);
    } else {
      onSearch(example);
    }
  };

  const toggleMode = () => {
    const newMode = mode === 'filter' ? 'semantic' : 'filter';
    setMode(newMode);
    setQuery(''); // Clear query when switching modes
  };

  const examples = mode === 'filter' ? FILTER_EXAMPLES : SEMANTIC_EXAMPLES;
  const defaultPlaceholder =
    mode === 'filter'
      ? 'Filter by keywords... (e.g., "high volatility", "bullish trend")'
      : 'Find similar periods... (e.g., "periods like current conditions")';

  return (
    <div className="space-y-3">
      {/* Mode Toggle */}
      {showModeToggle && onSemanticSearch && (
        <div className="flex items-center gap-2 mb-2">
          <span className="text-xs text-muted-foreground">Search mode:</span>
          <div className="flex rounded-lg border p-0.5 bg-muted/50">
            <button
              type="button"
              onClick={() => setMode('filter')}
              className={cn(
                'flex items-center gap-1.5 px-3 py-1 rounded-md text-xs font-medium transition-colors',
                mode === 'filter'
                  ? 'bg-background shadow-sm text-foreground'
                  : 'text-muted-foreground hover:text-foreground'
              )}
            >
              <Filter className="h-3 w-3" />
              Keyword Filter
            </button>
            <button
              type="button"
              onClick={() => setMode('semantic')}
              className={cn(
                'flex items-center gap-1.5 px-3 py-1 rounded-md text-xs font-medium transition-colors',
                mode === 'semantic'
                  ? 'bg-background shadow-sm text-foreground'
                  : 'text-muted-foreground hover:text-foreground'
              )}
            >
              <Brain className="h-3 w-3" />
              Semantic Search
            </button>
          </div>
          {mode === 'semantic' && (
            <Badge variant="secondary" className="text-xs">
              <Sparkles className="h-3 w-3 mr-1" />
              Vector Similarity
            </Badge>
          )}
        </div>
      )}

      {/* Search Input */}
      <form onSubmit={handleSubmit} className="flex gap-2">
        <div className="relative flex-1">
          {mode === 'semantic' ? (
            <Brain
              className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground"
            />
          ) : (
            <Search
              className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground"
            />
          )}
          <Input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder={placeholder || defaultPlaceholder}
            className="pl-10"
            disabled={isLoading}
          />
        </div>
        <Button
          type="submit"
          disabled={isLoading || !query.trim()}
          className={cn(mode === 'semantic' && 'bg-violet-600 hover:bg-violet-700')}
        >
          {isLoading ? 'Searching...' : mode === 'semantic' ? 'Find Similar' : 'Filter'}
        </Button>
        {query && (
          <Button
            type="button"
            variant="outline"
            onClick={() => {
              setQuery('');
              onSearch('');
            }}
          >
            Clear
          </Button>
        )}
      </form>

      {/* Example Queries */}
      <div className="flex items-center gap-2 flex-wrap">
        <span className="text-xs text-muted-foreground">
          {mode === 'filter' ? 'Filter by:' : 'Find periods like:'}
        </span>
        {examples.map((example) => (
          <button
            key={example}
            onClick={() => handleExampleClick(example)}
            className={cn(
              'text-xs px-2 py-1 rounded transition-colors',
              mode === 'semantic'
                ? 'bg-violet-500/10 hover:bg-violet-500/20 text-violet-700 dark:text-violet-300'
                : 'bg-muted hover:bg-muted/80'
            )}
            disabled={isLoading}
          >
            {example}
          </button>
        ))}
      </div>

      {/* Mode Description */}
      {showModeToggle && (
        <p className="text-xs text-muted-foreground">
          {mode === 'filter' ? (
            <>
              <strong>Keyword Filter:</strong> Searches by metadata fields like
              volatility, trend direction, and dates.
            </>
          ) : (
            <>
              <strong>Semantic Search:</strong> Uses 512-dimensional vector embeddings
              to find historically similar market conditions based on technical
              patterns, not just keywords.
            </>
          )}
        </p>
      )}
    </div>
  );
}
