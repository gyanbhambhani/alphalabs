'use client';

import { useState, useCallback, useRef, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Badge } from '@/components/ui/badge';
import { Skeleton } from '@/components/ui/skeleton';
import { 
  Search, 
  Loader2, 
  AlertCircle, 
  X, 
  Sparkles,
  TrendingUp,
  History,
  Shield,
  BarChart2,
} from 'lucide-react';
import { 
  useStreamingAnalysis, 
  useChunksByType 
} from '@/hooks/useStreamingAnalysis';
import { api } from '@/lib/api';
import {
  SimilarPeriodsChart,
  VolatilityRegimeChart,
  RiskMetricsTable,
  ReturnsDistributionChart,
  SharpeEvolutionChart,
} from '@/components/quant-charts';
import type { 
  ChartSpec, 
  TableSpec, 
  StreamChunk,
  SimilarPeriodsChartData,
  VolatilityRegimeChartData,
  ReturnsDistributionChartData,
  SharpeEvolutionChartData,
  RiskMetricsTableData,
} from '@/types';
import { cn } from '@/lib/utils';

const EXAMPLE_QUERIES = [
  { query: 'Is NVDA risky right now?', symbols: ['NVDA'] },
  { query: 'What happened after similar drops?', symbols: ['SPY'] },
  { query: 'Compare volatility', symbols: ['AAPL'] },
  { query: 'Should I buy TSLA?', symbols: ['TSLA'] },
  { query: 'MSFT performance analysis', symbols: ['MSFT'] },
];

const POPULAR_SYMBOLS = ['SPY', 'AAPL', 'NVDA', 'MSFT', 'TSLA', 'GOOGL', 'AMZN'];

interface SymbolSuggestion {
  symbol: string;
  name: string;
  sector: string;
}

export default function SearchPage() {
  const [query, setQuery] = useState('');
  const [symbols, setSymbols] = useState<string[]>([]);
  const [symbolInput, setSymbolInput] = useState('');
  const [suggestions, setSuggestions] = useState<SymbolSuggestion[]>([]);
  const [showSuggestions, setShowSuggestions] = useState(false);
  
  const inputRef = useRef<HTMLInputElement>(null);
  const suggestionsRef = useRef<HTMLDivElement>(null);
  
  const { 
    status, 
    chunks, 
    error, 
    analyze, 
    cancel, 
    reset 
  } = useStreamingAnalysis();
  
  const { textChunks, chartChunks, tableChunks } = useChunksByType(chunks);
  
  // Fetch symbol suggestions
  useEffect(() => {
    const fetchSuggestions = async () => {
      if (symbolInput.length < 1) {
        setSuggestions([]);
        return;
      }
      
      try {
        const result = await api.searchSymbols(symbolInput, 5);
        setSuggestions(result.symbols);
      } catch (err) {
        console.error('Failed to fetch suggestions:', err);
      }
    };
    
    const debounce = setTimeout(fetchSuggestions, 200);
    return () => clearTimeout(debounce);
  }, [symbolInput]);
  
  // Close suggestions when clicking outside
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (
        suggestionsRef.current && 
        !suggestionsRef.current.contains(e.target as Node)
      ) {
        setShowSuggestions(false);
      }
    };
    
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);
  
  const handleAddSymbol = useCallback((symbol: string) => {
    const upperSymbol = symbol.toUpperCase().trim();
    if (upperSymbol && !symbols.includes(upperSymbol)) {
      setSymbols(prev => [...prev, upperSymbol]);
    }
    setSymbolInput('');
    setShowSuggestions(false);
  }, [symbols]);
  
  const handleRemoveSymbol = useCallback((symbol: string) => {
    setSymbols(prev => prev.filter(s => s !== symbol));
  }, []);
  
  const handleSubmit = useCallback(async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!symbols.length) {
      return;
    }
    
    const searchQuery = query.trim() || 'Analyze this stock';
    await analyze(searchQuery, symbols);
  }, [query, symbols, analyze]);
  
  const handleExampleClick = useCallback((example: typeof EXAMPLE_QUERIES[0]) => {
    setQuery(example.query);
    setSymbols(example.symbols);
    analyze(example.query, example.symbols);
  }, [analyze]);
  
  const renderChart = useCallback((chunk: StreamChunk) => {
    const chartContent = chunk.content as ChartSpec;
    const { type, data, config } = chartContent;
    
    switch (type) {
      case 'similar_periods':
        return (
          <SimilarPeriodsChart 
            data={data as SimilarPeriodsChartData} 
            config={config} 
          />
        );
      case 'volatility_regime':
        return (
          <VolatilityRegimeChart 
            data={data as VolatilityRegimeChartData} 
            config={config} 
          />
        );
      case 'returns_distribution':
        return (
          <ReturnsDistributionChart 
            data={data as ReturnsDistributionChartData} 
            config={config} 
          />
        );
      case 'sharpe_evolution':
        return (
          <SharpeEvolutionChart 
            data={data as SharpeEvolutionChartData} 
            config={config} 
          />
        );
      default:
        return (
          <Card>
            <CardContent className="py-4">
              <p className="text-sm text-muted-foreground">
                Unknown chart type: {type}
              </p>
            </CardContent>
          </Card>
        );
    }
  }, []);
  
  const renderTable = useCallback((chunk: StreamChunk) => {
    const tableContent = chunk.content as TableSpec;
    return <RiskMetricsTable data={tableContent as RiskMetricsTableData} />;
  }, []);
  
  const isLoading = status === 'creating_session' || status === 'streaming';
  
  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold mb-2 flex items-center gap-2">
          <Search className="h-6 w-6" />
          AI Stock Terminal
        </h1>
        <p className="text-muted-foreground">
          Ask anything about stocks. Get instant analysis with charts and insights.
        </p>
      </div>
      
      {/* Search Form */}
      <Card>
        <CardContent className="pt-6">
          <form onSubmit={handleSubmit} className="space-y-4">
            {/* Symbol Input */}
            <div className="relative" ref={suggestionsRef}>
              <label className="text-sm font-medium mb-2 block">
                Stock Symbols
              </label>
              <div className="flex flex-wrap gap-2 mb-2">
                {symbols.map(symbol => (
                  <Badge 
                    key={symbol} 
                    variant="secondary"
                    className="px-3 py-1 text-sm"
                  >
                    {symbol}
                    <button
                      type="button"
                      onClick={() => handleRemoveSymbol(symbol)}
                      className="ml-2 hover:text-destructive"
                    >
                      <X className="h-3 w-3" />
                    </button>
                  </Badge>
                ))}
              </div>
              <Input
                ref={inputRef}
                value={symbolInput}
                onChange={(e) => {
                  setSymbolInput(e.target.value);
                  setShowSuggestions(true);
                }}
                onFocus={() => setShowSuggestions(true)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') {
                    e.preventDefault();
                    handleAddSymbol(symbolInput);
                  }
                }}
                placeholder="Add stock symbol (e.g., AAPL, NVDA)"
                className="max-w-xs"
                disabled={isLoading}
              />
              
              {/* Suggestions Dropdown */}
              {showSuggestions && suggestions.length > 0 && (
                <div className="absolute z-50 mt-1 w-72 bg-popover border rounded-md shadow-lg">
                  {suggestions.map(s => (
                    <button
                      key={s.symbol}
                      type="button"
                      className="w-full px-3 py-2 text-left hover:bg-muted flex justify-between"
                      onClick={() => handleAddSymbol(s.symbol)}
                    >
                      <span className="font-medium">{s.symbol}</span>
                      <span className="text-sm text-muted-foreground truncate ml-2">
                        {s.name}
                      </span>
                    </button>
                  ))}
                </div>
              )}
              
              {/* Popular Symbols */}
              {symbols.length === 0 && (
                <div className="flex flex-wrap gap-2 mt-2">
                  <span className="text-xs text-muted-foreground">Popular:</span>
                  {POPULAR_SYMBOLS.map(symbol => (
                    <button
                      key={symbol}
                      type="button"
                      onClick={() => handleAddSymbol(symbol)}
                      className="text-xs text-primary hover:underline"
                      disabled={isLoading}
                    >
                      {symbol}
                    </button>
                  ))}
                </div>
              )}
            </div>
            
            {/* Query Input */}
            <div>
              <label className="text-sm font-medium mb-2 block">
                Your Question
              </label>
              <div className="flex gap-2">
                <Input
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  placeholder="e.g., Is this stock risky? What happened after similar drops?"
                  className="flex-1"
                  disabled={isLoading}
                />
                <Button 
                  type="submit" 
                  disabled={!symbols.length || isLoading}
                >
                  {isLoading ? (
                    <>
                      <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                      Analyzing
                    </>
                  ) : (
                    <>
                      <Sparkles className="h-4 w-4 mr-2" />
                      Analyze
                    </>
                  )}
                </Button>
                {isLoading && (
                  <Button 
                    type="button" 
                    variant="outline" 
                    onClick={cancel}
                  >
                    Cancel
                  </Button>
                )}
              </div>
            </div>
          </form>
        </CardContent>
      </Card>
      
      {/* Example Queries */}
      {status === 'idle' && chunks.length === 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="text-sm">Try an example</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex flex-wrap gap-2">
              {EXAMPLE_QUERIES.map((example, idx) => (
                <Button
                  key={idx}
                  variant="outline"
                  size="sm"
                  onClick={() => handleExampleClick(example)}
                  className="text-xs"
                >
                  {example.query}
                  <Badge variant="secondary" className="ml-2">
                    {example.symbols.join(', ')}
                  </Badge>
                </Button>
              ))}
            </div>
          </CardContent>
        </Card>
      )}
      
      {/* Error Display */}
      {status === 'error' && error && (
        <Card className="border-destructive">
          <CardContent className="py-4 flex items-center gap-2 text-destructive">
            <AlertCircle className="h-5 w-5" />
            <span>{error}</span>
            <Button 
              variant="outline" 
              size="sm" 
              onClick={reset}
              className="ml-auto"
            >
              Try Again
            </Button>
          </CardContent>
        </Card>
      )}
      
      {/* Loading State */}
      {isLoading && chunks.length === 0 && (
        <div className="space-y-4">
          <Skeleton className="h-24 w-full" />
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <Skeleton className="h-64" />
            <Skeleton className="h-64" />
          </div>
        </div>
      )}
      
      {/* Results */}
      {chunks.length > 0 && (
        <div className="space-y-6">
          {/* Text Output */}
          {textChunks.length > 0 && (
            <Card>
              <CardContent className="py-4">
                <div className="space-y-2">
                  {textChunks.map((chunk, idx) => {
                    const text = chunk.content as string;
                    const isWarning = text.includes('⚠️');
                    const isAI = chunk.metadata?.stage === 'ai_synthesis';
                    
                    return (
                      <p 
                        key={idx} 
                        className={cn(
                          'text-sm',
                          isWarning && 'text-yellow-600',
                          isAI && 'italic text-muted-foreground'
                        )}
                      >
                        {text}
                      </p>
                    );
                  })}
                </div>
              </CardContent>
            </Card>
          )}
          
          {/* Charts */}
          {chartChunks.length > 0 && (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              {chartChunks.map((chunk, idx) => (
                <div key={idx}>
                  {renderChart(chunk)}
                </div>
              ))}
            </div>
          )}
          
          {/* Tables */}
          {tableChunks.length > 0 && (
            <div className="space-y-4">
              {tableChunks.map((chunk, idx) => (
                <div key={idx}>
                  {renderTable(chunk)}
                </div>
              ))}
            </div>
          )}
          
          {/* Analysis Complete */}
          {status === 'complete' && (
            <div className="flex justify-center">
              <Button 
                variant="outline" 
                onClick={reset}
                className="gap-2"
              >
                <Search className="h-4 w-4" />
                New Analysis
              </Button>
            </div>
          )}
        </div>
      )}
      
      {/* Feature Cards */}
      {status === 'idle' && chunks.length === 0 && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <FeatureCard
            icon={<History className="h-6 w-6" />}
            title="Historical Analogs"
            description="Find past periods similar to today using AI embeddings"
          />
          <FeatureCard
            icon={<TrendingUp className="h-6 w-6" />}
            title="Volatility Regimes"
            description="Detect market regime shifts with ML clustering"
          />
          <FeatureCard
            icon={<Shield className="h-6 w-6" />}
            title="Risk Metrics"
            description="VaR, CVaR, drawdowns, and risk analysis"
          />
          <FeatureCard
            icon={<BarChart2 className="h-6 w-6" />}
            title="Returns Analysis"
            description="Distribution stats, Sharpe ratio evolution"
          />
        </div>
      )}
    </div>
  );
}

function FeatureCard({ 
  icon, 
  title, 
  description 
}: { 
  icon: React.ReactNode; 
  title: string; 
  description: string;
}) {
  return (
    <Card className="hover:bg-muted/50 transition-colors">
      <CardContent className="pt-6">
        <div className="flex flex-col items-center text-center space-y-2">
          <div className="text-primary">{icon}</div>
          <h3 className="font-medium">{title}</h3>
          <p className="text-xs text-muted-foreground">{description}</p>
        </div>
      </CardContent>
    </Card>
  );
}
