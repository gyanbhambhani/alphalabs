'use client';

/**
 * Open-Source Bloomberg Terminal
 *
 * Real-time data + LLM-powered research. Fully free for students.
 * BYOK: Add your OpenAI API key to use AI analysis (stored in browser only).
 */

import { useState, useCallback, useEffect, useRef } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Badge } from '@/components/ui/badge';
import { Skeleton } from '@/components/ui/skeleton';
import {
  Search,
  Loader2,
  Key,
  TrendingUp,
  TrendingDown,
  RefreshCw,
  Sparkles,
} from 'lucide-react';
import {
  useStreamingAnalysis,
  useChunksByType,
} from '@/hooks/useStreamingAnalysis';
import { api } from '@/lib/api';
import {
  SimilarPeriodsChart,
  VolatilityRegimeChart,
  RiskMetricsTable,
  ReturnsDistributionChart,
} from '@/components/quant-charts';
import type {
  ChartSpec,
  StreamChunk,
  SimilarPeriodsChartData,
  VolatilityRegimeChartData,
  ReturnsDistributionChartData,
  RiskMetricsTableData,
} from '@/types';
import { cn } from '@/lib/utils';

const STORAGE_KEY = 'alphalabs_openai_key';
const TICKER_SYMBOLS = 'SPY,AAPL,MSFT,GOOGL,AMZN,NVDA,META,TSLA';

interface LivePrice {
  symbol: string;
  price: number;
  change: number;
  change_pct: number;
}

export default function TerminalPage() {
  const [apiKey, setApiKey] = useState('');
  const [showKeyModal, setShowKeyModal] = useState(false);
  const [livePrices, setLivePrices] = useState<LivePrice[]>([]);
  const [tickerLoading, setTickerLoading] = useState(true);
  const [query, setQuery] = useState('');
  const [symbols, setSymbols] = useState<string[]>(['SPY']);
  const [symbolInput, setSymbolInput] = useState('');

  const { status, chunks, error, analyze, cancel, reset } =
    useStreamingAnalysis();
  const { textChunks, chartChunks, tableChunks } = useChunksByType(chunks);

  // Load API key from localStorage
  useEffect(() => {
    try {
      const stored = localStorage.getItem(STORAGE_KEY);
      if (stored) setApiKey(stored);
    } catch {
      /* ignore */
    }
  }, []);

  const saveApiKey = useCallback(() => {
    const key = apiKey.trim();
    if (key.startsWith('sk-')) {
      try {
        localStorage.setItem(STORAGE_KEY, key);
        setShowKeyModal(false);
      } catch {
        /* ignore */
      }
    }
  }, [apiKey]);

  // Fetch live prices
  const fetchPrices = useCallback(async () => {
    setTickerLoading(true);
    try {
      const data = await api.getLivePrices(TICKER_SYMBOLS);
      setLivePrices(data.prices);
    } catch {
      setLivePrices([]);
    } finally {
      setTickerLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchPrices();
    const interval = setInterval(fetchPrices, 30000); // Refresh every 30s
    return () => clearInterval(interval);
  }, [fetchPrices]);

  const handleAnalyze = useCallback(
    async (q: string, syms: string[]) => {
      const key = apiKey.trim().startsWith('sk-') ? apiKey.trim() : undefined;
      await analyze(q || 'Analyze these stocks', syms, key);
    },
    [analyze, apiKey]
  );

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (symbols.length) handleAnalyze(query, symbols);
  };

  const addSymbol = (s: string) => {
    const upper = s.toUpperCase().trim();
    if (upper && !symbols.includes(upper)) {
      setSymbols((prev) => [...prev, upper]);
    }
    setSymbolInput('');
  };

  return (
    <div className="min-h-screen bg-[#0d1117] text-[#e6edf3]">
      {/* Ticker tape */}
      <div className="border-b border-[#30363d] bg-[#161b22] py-2 overflow-hidden">
        <div className="flex items-center gap-6 overflow-x-auto pb-1">
          {tickerLoading ? (
            <Skeleton className="h-6 w-48" />
          ) : (
            livePrices.map((p) => (
              <div key={p.symbol} className="flex items-center gap-2 shrink-0">
                <span className="font-mono font-semibold text-sm">
                  {p.symbol}
                </span>
                <span className="font-mono text-sm">${p.price.toFixed(2)}</span>
                <span
                  className={cn(
                    'font-mono text-xs',
                    p.change >= 0 ? 'text-emerald-400' : 'text-red-400'
                  )}
                >
                  {p.change >= 0 ? '+' : ''}
                  {p.change_pct.toFixed(2)}%
                </span>
              </div>
            ))
          )}
        </div>
        <p className="text-[10px] text-[#8b949e] mt-1 text-center">
          ~15 min delayed (Yahoo Finance free) · Refresh every 30s
        </p>
      </div>

      {/* Header */}
      <div className="border-b border-[#30363d] px-4 py-3 flex items-center justify-between">
        <div>
          <h1 className="text-xl font-bold">Terminal</h1>
          <p className="text-xs text-[#8b949e]">
            Open-source Bloomberg-style · BYOK for free AI analysis
          </p>
        </div>
        <Button
          variant="outline"
          size="sm"
          className="border-[#30363d] text-[#8b949e] hover:bg-[#21262d]"
          onClick={() => setShowKeyModal(true)}
        >
          <Key className="w-4 h-4 mr-2" />
          {apiKey ? 'Update API Key' : 'Add OpenAI Key'}
        </Button>
      </div>

      {/* Key modal */}
      {showKeyModal && (
        <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-50">
          <Card className="w-full max-w-md bg-[#161b22] border-[#30363d]">
            <CardHeader>
              <CardTitle className="text-[#e6edf3]">
                Bring Your Own Key (BYOK)
              </CardTitle>
              <p className="text-sm text-[#8b949e]">
                Add your OpenAI API key for free AI analysis. Stored in your
                browser only—never sent to our servers except per-request.
              </p>
            </CardHeader>
            <CardContent className="space-y-4">
              <Input
                type="password"
                placeholder="sk-..."
                value={apiKey}
                onChange={(e) => setApiKey(e.target.value)}
                className="bg-[#0d1117] border-[#30363d]"
              />
              <div className="flex gap-2">
                <Button onClick={saveApiKey} disabled={!apiKey.startsWith('sk-')}>
                  Save
                </Button>
                <Button
                  variant="outline"
                  onClick={() => setShowKeyModal(false)}
                  className="border-[#30363d]"
                >
                  Cancel
                </Button>
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Main content */}
      <div className="p-4 grid gap-4 lg:grid-cols-3">
        {/* Search panel */}
        <Card className="lg:col-span-1 bg-[#161b22] border-[#30363d]">
          <CardHeader>
            <CardTitle className="text-[#e6edf3] flex items-center gap-2">
              <Sparkles className="w-4 h-4" />
              Ask anything
            </CardTitle>
            <p className="text-xs text-[#8b949e]">
              {apiKey
                ? 'AI analysis enabled'
                : 'Add API key above for AI analysis'}
            </p>
          </CardHeader>
          <CardContent className="space-y-4">
            <form onSubmit={handleSubmit} className="space-y-3">
              <Input
                placeholder="e.g. Is NVDA risky? Compare AAPL to MSFT"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                className="bg-[#0d1117] border-[#30363d]"
              />
              <div className="flex flex-wrap gap-2">
                {symbols.map((s) => (
                  <Badge
                    key={s}
                    variant="secondary"
                    className="bg-[#21262d] cursor-pointer"
                    onClick={() =>
                      setSymbols((prev) => prev.filter((x) => x !== s))
                    }
                  >
                    {s} ×
                  </Badge>
                ))}
                <Input
                  placeholder="+ symbol"
                  value={symbolInput}
                  onChange={(e) => setSymbolInput(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter') {
                      e.preventDefault();
                      addSymbol(symbolInput);
                    }
                  }}
                  className="w-24 bg-[#0d1117] border-[#30363d]"
                />
              </div>
              <Button
                type="submit"
                disabled={!symbols.length || status === 'streaming'}
                className="w-full"
              >
                {status === 'streaming' ? (
                  <Loader2 className="w-4 h-4 animate-spin" />
                ) : (
                  <Search className="w-4 h-4" />
                )}
                {status === 'streaming' ? 'Analyzing...' : 'Analyze'}
              </Button>
            </form>
            <div className="flex flex-wrap gap-1">
              {['SPY', 'AAPL', 'NVDA', 'TSLA'].map((s) => (
                <Button
                  key={s}
                  variant="ghost"
                  size="sm"
                  className="text-xs text-[#8b949e]"
                  onClick={() => addSymbol(s)}
                >
                  {s}
                </Button>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Results */}
        <Card className="lg:col-span-2 bg-[#161b22] border-[#30363d] min-h-[400px]">
          <CardHeader className="flex flex-row items-center justify-between">
            <CardTitle className="text-[#e6edf3]">Analysis</CardTitle>
            {error && (
              <Badge variant="destructive" className="text-xs">
                {error}
              </Badge>
            )}
          </CardHeader>
          <CardContent className="space-y-4">
            {textChunks.map((c, i) => (
              <p key={i} className="text-sm text-[#e6edf3] whitespace-pre-wrap">
                {c.content as string}
              </p>
            ))}
            {chartChunks.map((c, i) => {
              const spec = c.content as ChartSpec;
              switch (spec.type) {
                case 'similar_periods':
                  return (
                    <SimilarPeriodsChart
                      key={i}
                      data={spec.data as unknown as SimilarPeriodsChartData}
                      config={spec.config}
                    />
                  );
                case 'volatility_regime':
                  return (
                    <VolatilityRegimeChart
                      key={i}
                      data={spec.data as unknown as VolatilityRegimeChartData}
                      config={spec.config}
                    />
                  );
                case 'returns_distribution':
                  return (
                    <ReturnsDistributionChart
                      key={i}
                      data={
                        spec.data as unknown as ReturnsDistributionChartData
                      }
                      config={spec.config}
                    />
                  );
                case 'risk_metrics':
                  return (
                    <RiskMetricsTable
                      key={i}
                      data={spec.data as unknown as RiskMetricsTableData}
                      config={spec.config}
                    />
                  );
                default:
                  return null;
              }
            })}
            {tableChunks.map((c, i) => {
              const t = c.content as { title?: string; columns?: string[]; rows?: Record<string, unknown>[] };
              return (
                <div key={i} className="overflow-x-auto">
                  {t.title && (
                    <p className="text-sm font-medium mb-2">{t.title}</p>
                  )}
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b border-[#30363d]">
                        {(t.columns || []).map((col) => (
                          <th
                            key={col}
                            className="text-left py-2 px-2 text-[#8b949e]"
                          >
                            {col}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {(t.rows || []).map((row, ri) => (
                        <tr
                          key={ri}
                          className="border-b border-[#21262d]"
                        >
                          {(t.columns || []).map((col) => (
                            <td key={col} className="py-2 px-2">
                              {String(row[col] ?? '')}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              );
            })}
            {status === 'idle' && !chunks.length && (
              <p className="text-[#8b949e] text-sm">
                Enter a query and symbols, then click Analyze. Add your OpenAI
                key for AI-powered insights.
              </p>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
