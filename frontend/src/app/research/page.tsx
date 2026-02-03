'use client';

import { useEffect, useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Badge } from '@/components/ui/badge';
import { SearchBar } from '@/components/SearchBar';
import { EmbeddingsTable } from '@/components/EmbeddingsTable';
import { EmbeddingsTimeline } from '@/components/EmbeddingsTimeline';
import { EmbeddingDetail } from '@/components/EmbeddingDetail';
import { StockSelector } from '@/components/StockSelector';
import { MultiStockCompare } from '@/components/MultiStockCompare';
import { api } from '@/lib/api';
import type {
  Embedding,
  EmbeddingsStats,
} from '@/types';

export default function ResearchPage() {
  const [selectedSymbol, setSelectedSymbol] = useState<string>('');
  const [selectedSymbols, setSelectedSymbols] = useState<string[]>([]);
  const [viewMode, setViewMode] = useState<'single' | 'compare'>('single');
  
  const [stats, setStats] = useState<EmbeddingsStats | null>(null);
  const [embeddings, setEmbeddings] = useState<Embedding[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [perPage] = useState(50);
  const [sortBy, setSortBy] = useState<
    'date' | 'return_1m' | 'volatility_21d' | 'price'
  >('date');
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('desc');
  const [isLoading, setIsLoading] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [searchInterpretation, setSearchInterpretation] = useState('');
  const [selectedEmbedding, setSelectedEmbedding] = useState<Embedding | null>(null);
  const [activeTab, setActiveTab] = useState<'table' | 'timeline'>('table');
  const [isLive, setIsLive] = useState(false);

  useEffect(() => {
    if (selectedSymbol) {
      loadStats();
    }
  }, [selectedSymbol]);

  useEffect(() => {
    if (selectedSymbol) {
      loadEmbeddings();
    }
  }, [selectedSymbol, page, sortBy, sortOrder, searchQuery]);

  const loadStats = async () => {
    if (!selectedSymbol) return;
    
    try {
      const data = await api.getEmbeddingsStatsForSymbol(selectedSymbol);
      setStats(data);
      setIsLive(true);
    } catch (error) {
      console.error('Failed to load stats:', error);
      setIsLive(false);
    }
  };

  const loadEmbeddings = async () => {
    if (!selectedSymbol) return;
    
    setIsLoading(true);
    try {
      if (searchQuery) {
        const result = await api.searchEmbeddingsForSymbol(
          selectedSymbol,
          searchQuery,
          perPage
        );
        setEmbeddings(result.results);
        setTotal(result.total);
        setSearchInterpretation(result.interpretation);
      } else {
        const result = await api.getEmbeddingsForSymbol(selectedSymbol, {
          page,
          perPage,
          sortBy,
          order: sortOrder,
        });
        setEmbeddings(result.embeddings);
        setTotal(result.total);
        setSearchInterpretation('');
      }
    } catch (error) {
      console.error('Failed to load embeddings:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSearch = (query: string) => {
    setSearchQuery(query);
    setPage(1);
  };

  const handleSemanticSearch = async (query: string) => {
    if (!selectedSymbol) return;
    
    setIsLoading(true);
    try {
      const result = await api.semanticSearch(selectedSymbol, query, perPage);
      
      // Convert semantic results to embedding format for display
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const semanticEmbeddings = result.results.map((r: any) => ({
        id: r.date,
        metadata: {
          date: r.metadata.date,
          return1w: 0,
          return1m: r.metadata.return_1m || 0,
          return3m: r.metadata.return_3m || 0,
          return6m: 0,
          return12m: 0,
          volatility5d: 0,
          volatility10d: 0,
          volatility21d: r.metadata.volatility_21d || 0,
          volatility63d: 0,
          price: r.metadata.price || 0,
        },
        similarity: r.similarity,
      }));
      
      setEmbeddings(semanticEmbeddings);
      setTotal(result.results.length);
      setSearchInterpretation(result.interpretation);
    } catch (error) {
      console.error('Semantic search failed:', error);
      setSearchInterpretation('Semantic search failed. Try keyword filter instead.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleSort = (field: string, order: 'asc' | 'desc') => {
    setSortBy(field as typeof sortBy);
    setSortOrder(order);
    setPage(1);
  };

  const handlePageChange = (newPage: number) => {
    setPage(newPage);
  };

  const handleRowClick = (embedding: Embedding) => {
    setSelectedEmbedding(embedding);
  };

  const handleFindSimilar = (embedding: Embedding) => {
    setSearchQuery(embedding.metadata.date);
    setSelectedEmbedding(null);
  };

  const handleStockSelect = (symbol: string) => {
    setSelectedSymbol(symbol);
    setViewMode('single');
    setPage(1);
  };

  const handleMultipleSelect = (symbols: string[]) => {
    setSelectedSymbols(symbols);
    if (symbols.length > 1) {
      setViewMode('compare');
    } else if (symbols.length === 1) {
      setSelectedSymbol(symbols[0]);
      setViewMode('single');
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold mb-2">Market Research</h1>
        <p className="text-muted-foreground">
          Semantic market memory - explore historical market states and find similar periods
        </p>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm text-muted-foreground">
              Selected Stock
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-mono font-bold">
              {selectedSymbol || '-'}
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm text-muted-foreground">
              Data Points
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-mono font-bold">
              {stats ? stats.totalCount.toLocaleString() : '...'}
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm text-muted-foreground">
              Date Range
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-sm font-mono">
              {stats && stats.dateRange[0]
                ? `${stats.dateRange[0]} to ${stats.dateRange[1]}`
                : '...'}
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm text-muted-foreground">
              Status
            </CardTitle>
          </CardHeader>
          <CardContent>
            <Badge
              variant="outline"
              className={
                isLive
                  ? 'bg-green-500/20 text-green-500'
                  : 'bg-red-500/20 text-red-500'
              }
            >
              {isLive ? '● Connected' : '○ Disconnected'}
            </Badge>
          </CardContent>
        </Card>
      </div>

      {/* Main Content */}
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Stock Selector */}
        <div className="lg:col-span-1">
          <Tabs value={viewMode} onValueChange={(v) => setViewMode(v as typeof viewMode)}>
            <TabsList className="grid w-full grid-cols-2 mb-4">
              <TabsTrigger value="single">Single</TabsTrigger>
              <TabsTrigger value="compare">Compare</TabsTrigger>
            </TabsList>

            <TabsContent value="single">
              <StockSelector
                onSelectStock={handleStockSelect}
                selectedSymbol={selectedSymbol}
              />
            </TabsContent>

            <TabsContent value="compare">
              <StockSelector
                onSelectStock={() => {}}
                onSelectMultiple={handleMultipleSelect}
                selectedSymbols={selectedSymbols}
                multiSelect={true}
              />
            </TabsContent>
          </Tabs>
        </div>

        {/* Main Content Area */}
        <div className="lg:col-span-3 space-y-6">
          {viewMode === 'compare' && selectedSymbols.length > 0 ? (
            <MultiStockCompare symbols={selectedSymbols} />
          ) : selectedSymbol ? (
            <>
              {/* Search Bar */}
              <Card>
                <CardHeader>
                  <CardTitle>Semantic Market Search</CardTitle>
                </CardHeader>
                <CardContent>
                  <SearchBar
                    onSearch={handleSearch}
                    onSemanticSearch={handleSemanticSearch}
                    isLoading={isLoading}
                    showModeToggle={true}
                  />
                  {searchInterpretation && (
                    <div className="mt-3 p-3 bg-muted rounded-lg">
                      <p className="text-sm">
                        <span className="font-semibold">Search interpretation:</span>{' '}
                        {searchInterpretation}
                      </p>
                    </div>
                  )}
                </CardContent>
              </Card>

              {/* Data Display */}
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                <div className="lg:col-span-2">
                  <Card>
                    <CardHeader>
                      <div className="flex items-center justify-between">
                        <CardTitle>Market State Embeddings</CardTitle>
                        <div className="text-sm text-muted-foreground">
                          {total.toLocaleString()} results
                        </div>
                      </div>
                    </CardHeader>
                    <CardContent>
                      <Tabs
                        value={activeTab}
                        onValueChange={(v) => setActiveTab(v as typeof activeTab)}
                      >
                        <TabsList className="grid w-full grid-cols-2">
                          <TabsTrigger value="table">Table View</TabsTrigger>
                          <TabsTrigger value="timeline">Timeline View</TabsTrigger>
                        </TabsList>

                        <TabsContent value="table" className="mt-4">
                          <EmbeddingsTable
                            embeddings={embeddings}
                            total={total}
                            page={page}
                            perPage={perPage}
                            onPageChange={handlePageChange}
                            onSort={handleSort}
                            onRowClick={handleRowClick}
                            sortBy={sortBy}
                            sortOrder={sortOrder}
                            isLoading={isLoading}
                          />
                        </TabsContent>

                        <TabsContent value="timeline" className="mt-4">
                          <EmbeddingsTimeline
                            embeddings={embeddings}
                            onPointClick={handleRowClick}
                          />
                        </TabsContent>
                      </Tabs>
                    </CardContent>
                  </Card>
                </div>

                {/* Detail Panel */}
                <div>
                  {selectedEmbedding ? (
                    <EmbeddingDetail
                      embedding={selectedEmbedding}
                      onClose={() => setSelectedEmbedding(null)}
                      onFindSimilar={handleFindSimilar}
                    />
                  ) : (
                    <Card>
                      <CardHeader>
                        <CardTitle>Details</CardTitle>
                      </CardHeader>
                      <CardContent>
                        <p className="text-sm text-muted-foreground">
                          Click on any row in the table or point in the timeline to 
                          view detailed market state information.
                        </p>
                      </CardContent>
                    </Card>
                  )}
                </div>
              </div>
            </>
          ) : (
            <Card>
              <CardContent className="py-12">
                <div className="text-center text-muted-foreground">
                  <p className="text-lg mb-2">Select a stock from the sidebar</p>
                  <p className="text-sm">
                    Explore historical market states using semantic embeddings.
                    Find similar periods and understand what happened next.
                  </p>
                </div>
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}
