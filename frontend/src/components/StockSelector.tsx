'use client';

import { useEffect, useState } from 'react';
import { Search, Check, X } from 'lucide-react';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import type { Stock } from '@/types';
import { cn } from '@/lib/utils';

interface StockSelectorProps {
  onSelectStock: (symbol: string) => void;
  onSelectMultiple?: (symbols: string[]) => void;
  selectedSymbol?: string;
  selectedSymbols?: string[];
  multiSelect?: boolean;
}

export function StockSelector({
  onSelectStock,
  onSelectMultiple,
  selectedSymbol,
  selectedSymbols = [],
  multiSelect = false,
}: StockSelectorProps) {
  const [stocks, setStocks] = useState<Stock[]>([]);
  const [filteredStocks, setFilteredStocks] = useState<Stock[]>([]);
  const [sectors, setSectors] = useState<string[]>([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedSector, setSelectedSector] = useState<string>('');
  const [showEmbeddingsOnly, setShowEmbeddingsOnly] = useState(true);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    loadStocks();
  }, [showEmbeddingsOnly, selectedSector]);

  useEffect(() => {
    filterStocks();
  }, [searchQuery, stocks]);

  const loadStocks = async () => {
    setIsLoading(true);
    try {
      const { api } = await import('@/lib/api');
      const response = await api.getStocks({
        hasEmbeddings: showEmbeddingsOnly,
        sector: selectedSector || undefined,
      });
      
      setStocks(response.stocks);
      setSectors(response.sectors);
      setFilteredStocks(response.stocks);
    } catch (error) {
      console.error('Failed to load stocks:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const filterStocks = () => {
    if (!searchQuery.trim()) {
      setFilteredStocks(stocks);
      return;
    }

    const query = searchQuery.toLowerCase();
    const filtered = stocks.filter(
      (stock) =>
        stock.symbol.toLowerCase().includes(query) ||
        stock.name.toLowerCase().includes(query)
    );
    setFilteredStocks(filtered);
  };

  const handleStockClick = (symbol: string) => {
    if (multiSelect) {
      const newSelection = selectedSymbols.includes(symbol)
        ? selectedSymbols.filter((s) => s !== symbol)
        : [...selectedSymbols, symbol];
      onSelectMultiple?.(newSelection);
    } else {
      onSelectStock(symbol);
    }
  };

  const isSelected = (symbol: string) => {
    if (multiSelect) {
      return selectedSymbols.includes(symbol);
    }
    return selectedSymbol === symbol;
  };

  const clearSelection = () => {
    if (multiSelect) {
      onSelectMultiple?.([]);
    }
  };

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle>Select Stock{multiSelect && 's'}</CardTitle>
          {multiSelect && selectedSymbols.length > 0 && (
            <Button
              variant="outline"
              size="sm"
              onClick={clearSelection}
            >
              Clear ({selectedSymbols.length})
            </Button>
          )}
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Search */}
        <div className="relative">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <Input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search by symbol or name..."
            className="pl-10"
          />
        </div>

        {/* Filters */}
        <div className="flex flex-wrap gap-2">
          <Button
            variant={showEmbeddingsOnly ? 'default' : 'outline'}
            size="sm"
            onClick={() => setShowEmbeddingsOnly(!showEmbeddingsOnly)}
          >
            {showEmbeddingsOnly ? 'With Data' : 'All Stocks'}
          </Button>

          <select
            value={selectedSector}
            onChange={(e) => setSelectedSector(e.target.value)}
            className="text-sm rounded-md border bg-background px-3 py-1"
          >
            <option value="">All Sectors</option>
            {sectors.map((sector) => (
              <option key={sector} value={sector}>
                {sector}
              </option>
            ))}
          </select>
        </div>

        {/* Stock List */}
        <div className="max-h-96 overflow-y-auto space-y-1">
          {isLoading ? (
            <div className="text-center py-8 text-muted-foreground">
              Loading stocks...
            </div>
          ) : filteredStocks.length === 0 ? (
            <div className="text-center py-8 text-muted-foreground">
              No stocks found
            </div>
          ) : (
            filteredStocks.map((stock) => (
              <button
                key={stock.symbol}
                onClick={() => handleStockClick(stock.symbol)}
                className={cn(
                  'w-full text-left px-3 py-2 rounded-md',
                  'hover:bg-muted transition-colors',
                  'flex items-center justify-between',
                  isSelected(stock.symbol) && 'bg-primary/10 border border-primary'
                )}
              >
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2">
                    <span className="font-mono font-semibold">
                      {stock.symbol}
                    </span>
                    {stock.hasEmbeddings && (
                      <Badge variant="outline" className="text-xs">
                        {stock.embeddingsCount.toLocaleString()} records
                      </Badge>
                    )}
                  </div>
                  <div className="text-sm text-muted-foreground truncate">
                    {stock.name}
                  </div>
                  {stock.sector && (
                    <div className="text-xs text-muted-foreground">
                      {stock.sector}
                    </div>
                  )}
                </div>
                {isSelected(stock.symbol) && (
                  <Check className="h-4 w-4 text-primary flex-shrink-0 ml-2" />
                )}
              </button>
            ))
          )}
        </div>

        {/* Selection Info */}
        {multiSelect && selectedSymbols.length > 0 && (
          <div className="pt-2 border-t">
            <div className="text-sm text-muted-foreground mb-2">
              Selected: {selectedSymbols.length} stock{selectedSymbols.length !== 1 && 's'}
            </div>
            <div className="flex flex-wrap gap-1">
              {selectedSymbols.map((symbol) => (
                <Badge
                  key={symbol}
                  variant="secondary"
                  className="cursor-pointer"
                  onClick={() => handleStockClick(symbol)}
                >
                  {symbol}
                  <X className="h-3 w-3 ml-1" />
                </Badge>
              ))}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
