'use client';

import { useState } from 'react';
import { Search } from 'lucide-react';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';

interface SearchBarProps {
  onSearch: (query: string) => void;
  isLoading?: boolean;
  placeholder?: string;
}

const EXAMPLE_QUERIES = [
  'high volatility',
  'bullish trend',
  'market crash',
  'low volatility',
  'rally',
  'downtrend',
  '2023-11',
];

export function SearchBar({ 
  onSearch, 
  isLoading = false,
  placeholder = 'Search market conditions... (e.g., "high volatility", "bullish trend", "2023-11")'
}: SearchBarProps) {
  const [query, setQuery] = useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSearch(query);
  };

  const handleExampleClick = (example: string) => {
    setQuery(example);
    onSearch(example);
  };

  return (
    <div className="space-y-3">
      <form onSubmit={handleSubmit} className="flex gap-2">
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <Input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder={placeholder}
            className="pl-10"
            disabled={isLoading}
          />
        </div>
        <Button 
          type="submit" 
          disabled={isLoading || !query.trim()}
        >
          {isLoading ? 'Searching...' : 'Search'}
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

      <div className="flex items-center gap-2 flex-wrap">
        <span className="text-xs text-muted-foreground">Try:</span>
        {EXAMPLE_QUERIES.map((example) => (
          <button
            key={example}
            onClick={() => handleExampleClick(example)}
            className="text-xs px-2 py-1 rounded bg-muted 
                       hover:bg-muted/80 transition-colors"
            disabled={isLoading}
          >
            {example}
          </button>
        ))}
      </div>
    </div>
  );
}
