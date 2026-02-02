'use client';

import { useEffect, useState, useCallback } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { ChatPanel } from '@/components/ChatPanel';
import { StockSelector } from '@/components/StockSelector';
import {
  Search,
  TrendingUp,
  TrendingDown,
  Activity,
  Brain,
  RefreshCw,
  Zap,
} from 'lucide-react';
import { api } from '@/lib/api';
import { cn } from '@/lib/utils';

interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
  timestamp: string;
  queryType?: string;
  data?: Record<string, unknown>;
  suggestions?: string[];
}

interface SemanticResult {
  date: string;
  similarity: number;
  metadata: {
    date: string;
    return_1m: number;
    return_3m?: number;
    volatility_21d: number;
    price: number;
  };
  forward_return_5d?: number;
  forward_return_20d?: number;
}

interface SemanticSearchResponse {
  results: SemanticResult[];
  interpretation: string;
  avg_forward_return: number;
  positive_rate: number;
  current_state?: {
    date: string;
    volatility_21d: number;
    return_1m: number;
  };
}

const QUICK_QUERIES = [
  'Find periods similar to current conditions',
  'Show me historical crashes',
  'High volatility periods',
  'Strong rally periods',
  'What happened after similar setups?',
];

export default function LabPage() {
  const [activeTab, setActiveTab] = useState<'chat' | 'semantic'>('semantic');
  
  // Chat state
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [suggestions, setSuggestions] = useState<string[]>([]);
  const [isChatLoading, setIsChatLoading] = useState(false);
  const [conversationId] = useState(() => `conv_${Date.now()}`);
  
  // Semantic search state
  const [selectedSymbol, setSelectedSymbol] = useState<string>('SPY');
  const [searchQuery, setSearchQuery] = useState('');
  const [semanticResults, setSemanticResults] = useState<SemanticSearchResponse | null>(
    null
  );
  const [isSearching, setIsSearching] = useState(false);
  const [searchError, setSearchError] = useState<string | null>(null);

  useEffect(() => {
    loadSuggestions();
  }, []);

  const loadSuggestions = async () => {
    try {
      const data = await api.getLabSuggestions();
      setSuggestions(data.suggestions || QUICK_QUERIES);
    } catch (error) {
      console.error('Failed to load suggestions:', error);
      setSuggestions(QUICK_QUERIES);
    }
  };

  const handleChatMessage = useCallback(async (message: string) => {
    const userMessage: ChatMessage = {
      role: 'user',
      content: message,
      timestamp: new Date().toISOString(),
    };
    
    setMessages((prev) => [...prev, userMessage]);
    setIsChatLoading(true);

    try {
      const response = await api.labChat(message, conversationId);
      
      const assistantMessage: ChatMessage = {
        role: 'assistant',
        content: response.message,
        timestamp: new Date().toISOString(),
        queryType: response.query_type,
        data: response.data,
        suggestions: response.suggestions,
      };
      
      setMessages((prev) => [...prev, assistantMessage]);
      
      if (response.suggestions?.length > 0) {
        setSuggestions(response.suggestions);
      }
    } catch (error) {
      const errorMessage: ChatMessage = {
        role: 'assistant',
        content: `Sorry, I encountered an error: ${
          error instanceof Error ? error.message : 'Unknown error'
        }`,
        timestamp: new Date().toISOString(),
        queryType: 'error',
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsChatLoading(false);
    }
  }, [conversationId]);

  const handleSemanticSearch = useCallback(async (query?: string) => {
    const searchText = query || searchQuery || 'Find periods similar to current';
    
    if (!selectedSymbol) {
      setSearchError('Please select a stock first');
      return;
    }

    setIsSearching(true);
    setSearchError(null);

    try {
      const response = await api.semanticSearch(selectedSymbol, searchText, 20);
      setSemanticResults(response);
    } catch (error) {
      console.error('Semantic search failed:', error);
      setSearchError(
        error instanceof Error ? error.message : 'Search failed'
      );
      setSemanticResults(null);
    } finally {
      setIsSearching(false);
    }
  }, [selectedSymbol, searchQuery]);

  const handleStockSelect = (symbol: string) => {
    setSelectedSymbol(symbol);
    setSemanticResults(null);
    setSearchError(null);
  };

  const formatPercent = (value: number | null | undefined): string => {
    if (value === null || value === undefined) return 'N/A';
    return `${(value * 100).toFixed(1)}%`;
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold mb-2">AI Trading Lab</h1>
        <p className="text-muted-foreground">
          Semantic market analysis powered by vector embeddings
        </p>
      </div>

      {/* Mode Tabs */}
      <Tabs
        value={activeTab}
        onValueChange={(v) => setActiveTab(v as typeof activeTab)}
      >
        <TabsList className="grid w-full max-w-md grid-cols-2">
          <TabsTrigger value="semantic" className="flex items-center gap-2">
            <Brain className="h-4 w-4" />
            Semantic Search
          </TabsTrigger>
          <TabsTrigger value="chat" className="flex items-center gap-2">
            <Zap className="h-4 w-4" />
            AI Chat
          </TabsTrigger>
        </TabsList>

        {/* Semantic Search Tab */}
        <TabsContent value="semantic" className="mt-6">
          <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
            {/* Stock Selector */}
            <div className="lg:col-span-1">
              <StockSelector
                onSelectStock={handleStockSelect}
                selectedSymbol={selectedSymbol}
              />
            </div>

            {/* Search & Results */}
            <div className="lg:col-span-3 space-y-6">
              {/* Search Input */}
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Search className="h-5 w-5" />
                    Vector Similarity Search
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <form
                    onSubmit={(e) => {
                      e.preventDefault();
                      handleSemanticSearch();
                    }}
                    className="flex gap-2"
                  >
                    <Input
                      value={searchQuery}
                      onChange={(e) => setSearchQuery(e.target.value)}
                      placeholder="Describe market conditions to find... (or leave empty for current)"
                      className="flex-1"
                      disabled={isSearching}
                    />
                    <Button type="submit" disabled={isSearching || !selectedSymbol}>
                      {isSearching ? (
                        <RefreshCw className="h-4 w-4 animate-spin" />
                      ) : (
                        <Search className="h-4 w-4" />
                      )}
                    </Button>
                  </form>

                  {/* Quick Queries */}
                  <div className="flex flex-wrap gap-2">
                    {QUICK_QUERIES.map((query) => (
                      <Button
                        key={query}
                        variant="outline"
                        size="sm"
                        className="text-xs"
                        onClick={() => {
                          setSearchQuery(query);
                          handleSemanticSearch(query);
                        }}
                        disabled={isSearching || !selectedSymbol}
                      >
                        {query}
                      </Button>
                    ))}
                  </div>
                </CardContent>
              </Card>

              {/* Error Display */}
              {searchError && (
                <Card className="border-destructive">
                  <CardContent className="py-4">
                    <p className="text-destructive text-sm">{searchError}</p>
                  </CardContent>
                </Card>
              )}

              {/* Results */}
              {semanticResults && (
                <>
                  {/* Summary Cards */}
                  <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                    <Card>
                      <CardHeader className="pb-2">
                        <CardTitle className="text-sm text-muted-foreground">
                          Similar Periods
                        </CardTitle>
                      </CardHeader>
                      <CardContent>
                        <p className="text-2xl font-bold">
                          {semanticResults.results.length}
                        </p>
                      </CardContent>
                    </Card>

                    <Card>
                      <CardHeader className="pb-2">
                        <CardTitle className="text-sm text-muted-foreground">
                          Avg Forward Return
                        </CardTitle>
                      </CardHeader>
                      <CardContent>
                        <p
                          className={cn(
                            'text-2xl font-bold',
                            semanticResults.avg_forward_return > 0
                              ? 'text-green-500'
                              : 'text-red-500'
                          )}
                        >
                          {formatPercent(semanticResults.avg_forward_return)}
                        </p>
                      </CardContent>
                    </Card>

                    <Card>
                      <CardHeader className="pb-2">
                        <CardTitle className="text-sm text-muted-foreground">
                          Positive Rate
                        </CardTitle>
                      </CardHeader>
                      <CardContent>
                        <p className="text-2xl font-bold">
                          {formatPercent(semanticResults.positive_rate)}
                        </p>
                      </CardContent>
                    </Card>

                    <Card>
                      <CardHeader className="pb-2">
                        <CardTitle className="text-sm text-muted-foreground">
                          Current Vol
                        </CardTitle>
                      </CardHeader>
                      <CardContent>
                        <p className="text-2xl font-bold font-mono">
                          {semanticResults.current_state
                            ? formatPercent(
                                semanticResults.current_state.volatility_21d
                              )
                            : 'N/A'}
                        </p>
                      </CardContent>
                    </Card>
                  </div>

                  {/* Interpretation */}
                  <Card>
                    <CardHeader>
                      <CardTitle className="flex items-center gap-2">
                        <Activity className="h-5 w-5" />
                        Analysis
                      </CardTitle>
                    </CardHeader>
                    <CardContent>
                      <p className="text-sm">{semanticResults.interpretation}</p>
                    </CardContent>
                  </Card>

                  {/* Results Table */}
                  <Card>
                    <CardHeader>
                      <CardTitle>Similar Historical Periods</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="overflow-x-auto">
                        <table className="w-full text-sm">
                          <thead>
                            <tr className="border-b">
                              <th className="text-left py-2 px-2">Date</th>
                              <th className="text-right py-2 px-2">Similarity</th>
                              <th className="text-right py-2 px-2">1M Return</th>
                              <th className="text-right py-2 px-2">Volatility</th>
                              <th className="text-right py-2 px-2">Price</th>
                              <th className="text-right py-2 px-2">Fwd 5D</th>
                              <th className="text-right py-2 px-2">Fwd 20D</th>
                            </tr>
                          </thead>
                          <tbody>
                            {semanticResults.results.map((result, idx) => (
                              <tr
                                key={result.date}
                                className={cn(
                                  'border-b hover:bg-muted/50',
                                  idx < 3 && 'bg-primary/5'
                                )}
                              >
                                <td className="py-2 px-2 font-mono">
                                  {result.date}
                                </td>
                                <td className="text-right py-2 px-2">
                                  <Badge
                                    variant="outline"
                                    className={cn(
                                      result.similarity > 0.9
                                        ? 'bg-green-500/20'
                                        : result.similarity > 0.8
                                        ? 'bg-yellow-500/20'
                                        : ''
                                    )}
                                  >
                                    {(result.similarity * 100).toFixed(1)}%
                                  </Badge>
                                </td>
                                <td
                                  className={cn(
                                    'text-right py-2 px-2 font-mono',
                                    result.metadata.return_1m > 0
                                      ? 'text-green-500'
                                      : 'text-red-500'
                                  )}
                                >
                                  {formatPercent(result.metadata.return_1m)}
                                </td>
                                <td className="text-right py-2 px-2 font-mono">
                                  {formatPercent(result.metadata.volatility_21d)}
                                </td>
                                <td className="text-right py-2 px-2 font-mono">
                                  ${result.metadata.price.toFixed(2)}
                                </td>
                                <td
                                  className={cn(
                                    'text-right py-2 px-2 font-mono',
                                    result.forward_return_5d === null ||
                                      result.forward_return_5d === undefined
                                      ? 'text-muted-foreground'
                                      : result.forward_return_5d > 0
                                      ? 'text-green-500'
                                      : 'text-red-500'
                                  )}
                                >
                                  {result.forward_return_5d !== null &&
                                  result.forward_return_5d !== undefined ? (
                                    <span className="flex items-center justify-end gap-1">
                                      {result.forward_return_5d > 0 ? (
                                        <TrendingUp className="h-3 w-3" />
                                      ) : (
                                        <TrendingDown className="h-3 w-3" />
                                      )}
                                      {formatPercent(result.forward_return_5d)}
                                    </span>
                                  ) : (
                                    'N/A'
                                  )}
                                </td>
                                <td
                                  className={cn(
                                    'text-right py-2 px-2 font-mono',
                                    result.forward_return_20d === null ||
                                      result.forward_return_20d === undefined
                                      ? 'text-muted-foreground'
                                      : result.forward_return_20d > 0
                                      ? 'text-green-500'
                                      : 'text-red-500'
                                  )}
                                >
                                  {result.forward_return_20d !== null &&
                                  result.forward_return_20d !== undefined ? (
                                    <span className="flex items-center justify-end gap-1">
                                      {result.forward_return_20d > 0 ? (
                                        <TrendingUp className="h-3 w-3" />
                                      ) : (
                                        <TrendingDown className="h-3 w-3" />
                                      )}
                                      {formatPercent(result.forward_return_20d)}
                                    </span>
                                  ) : (
                                    'N/A'
                                  )}
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </CardContent>
                  </Card>
                </>
              )}

              {/* Empty State */}
              {!semanticResults && !isSearching && !searchError && (
                <Card>
                  <CardContent className="py-12 text-center">
                    <Brain className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
                    <h3 className="text-lg font-medium mb-2">
                      Semantic Vector Search
                    </h3>
                    <p className="text-sm text-muted-foreground max-w-md mx-auto">
                      Find historical periods with similar market conditions using
                      512-dimensional embeddings. Select a stock and search to see
                      periods that match current conditions or your query.
                    </p>
                  </CardContent>
                </Card>
              )}
            </div>
          </div>
        </TabsContent>

        {/* Chat Tab */}
        <TabsContent value="chat" className="mt-6">
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="lg:col-span-2">
              <Card className="h-[600px]">
                <CardHeader>
                  <CardTitle>AI Research Assistant</CardTitle>
                </CardHeader>
                <CardContent className="h-[calc(100%-4rem)]">
                  <ChatPanel
                    messages={messages}
                    onSendMessage={handleChatMessage}
                    onSuggestionClick={handleChatMessage}
                    suggestions={suggestions}
                    isLoading={isChatLoading}
                  />
                </CardContent>
              </Card>
            </div>

            <div className="space-y-4">
              <Card>
                <CardHeader>
                  <CardTitle className="text-sm">Quick Actions</CardTitle>
                </CardHeader>
                <CardContent className="space-y-2">
                  <Button
                    variant="outline"
                    className="w-full justify-start"
                    onClick={() => handleChatMessage('What is the current market outlook?')}
                  >
                    Market Outlook
                  </Button>
                  <Button
                    variant="outline"
                    className="w-full justify-start"
                    onClick={() => handleChatMessage('Find periods similar to current conditions')}
                  >
                    Similar Periods
                  </Button>
                  <Button
                    variant="outline"
                    className="w-full justify-start"
                    onClick={() => handleChatMessage('What are the key risks right now?')}
                  >
                    Risk Analysis
                  </Button>
                  <Button
                    variant="outline"
                    className="w-full justify-start"
                    onClick={() => handleChatMessage('Generate a research report on SPY')}
                  >
                    SPY Research
                  </Button>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle className="text-sm">About</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-xs text-muted-foreground">
                    The AI assistant has access to historical market data,
                    semantic embeddings, and can generate research reports.
                    Ask about specific stocks, market conditions, or historical
                    patterns.
                  </p>
                </CardContent>
              </Card>
            </div>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
}
