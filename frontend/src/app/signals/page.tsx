'use client';

import { useEffect, useState, useCallback } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Skeleton } from '@/components/ui/skeleton';
import type { StrategySignals } from '@/types';
import { cn } from '@/lib/utils';
import { api } from '@/lib/api';

function getRegimeColor(regime: string): string {
  if (regime.includes('trending_up')) return 'bg-green-500/20 text-green-500';
  if (regime.includes('trending_down')) return 'bg-red-500/20 text-red-500';
  return 'bg-yellow-500/20 text-yellow-500';
}

function getRegimeDescription(regime: string): string {
  if (regime.includes('low_vol')) {
    if (regime.includes('trending_up')) {
      return 'Low volatility with upward momentum - favorable for trend following.';
    }
    if (regime.includes('trending_down')) {
      return 'Low volatility with downward pressure - watch for breakdown.';
    }
    return 'Low volatility, ranging market - consider mean reversion.';
  }
  if (regime.includes('high_vol')) {
    return 'High volatility environment - caution advised, mean reversion may work better.';
  }
  if (regime.includes('normal_vol')) {
    return 'Normal volatility - balanced approach recommended.';
  }
  return 'Market conditions being analyzed...';
}

function getFreshnessColor(freshness: string | undefined): string {
  if (!freshness) return 'bg-gray-500/20 text-gray-500';
  if (freshness === 'live') return 'bg-green-500/20 text-green-500';
  if (freshness === 'recent') return 'bg-blue-500/20 text-blue-500';
  if (freshness === 'error') return 'bg-red-500/20 text-red-500';
  return 'bg-yellow-500/20 text-yellow-500';
}

export default function SignalsPage() {
  const [signals, setSignals] = useState<StrategySignals | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lastRefresh, setLastRefresh] = useState<Date | null>(null);

  const loadSignals = useCallback(async (forceRefresh = false) => {
    setIsLoading(true);
    setError(null);
    try {
      // Add force_refresh query param if needed
      const endpoint = forceRefresh ? '/api/signals?force_refresh=true' : '/api/signals';
      const response = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}${endpoint}`
      );
      if (!response.ok) throw new Error('Failed to fetch signals');
      const data = await response.json();
      setSignals(data);
      setLastRefresh(new Date());
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load signals');
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    loadSignals();
    // Refresh every 5 minutes
    const interval = setInterval(() => loadSignals(), 5 * 60 * 1000);
    return () => clearInterval(interval);
  }, [loadSignals]);

  if (isLoading && !signals) {
    return (
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <Skeleton className="h-8 w-48 mb-2" />
            <Skeleton className="h-4 w-64" />
          </div>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {[1, 2, 3, 4, 5, 6].map(i => (
            <Card key={i}>
              <CardContent className="p-6">
                <Skeleton className="h-40 w-full" />
              </CardContent>
            </Card>
          ))}
        </div>
      </div>
    );
  }

  if (error && !signals) {
    return (
      <div className="space-y-6">
        <Card>
          <CardContent className="py-12 text-center">
            <p className="text-red-500 mb-4">{error}</p>
            <Button onClick={() => loadSignals()}>Retry</Button>
          </CardContent>
        </Card>
      </div>
    );
  }

  if (!signals) return null;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-2xl font-bold mb-2">Strategy Signals</h1>
          <p className="text-muted-foreground">
            Real-time signals computed from live market data via yfinance
          </p>
        </div>
        <div className="flex items-center gap-3">
          <Badge 
            variant="outline" 
            className={getFreshnessColor(signals.dataFreshness)}
          >
            {signals.dataFreshness === 'live' && '● Live Data'}
            {signals.dataFreshness === 'recent' && '● Recent'}
            {signals.dataFreshness === 'error' && '○ Error'}
            {signals.dataFreshness && !['live', 'recent', 'error'].includes(signals.dataFreshness) && 
              `⏱ ${signals.dataFreshness}`}
            {!signals.dataFreshness && '○ Unknown'}
          </Badge>
          <Button 
            variant="outline" 
            size="sm"
            onClick={() => loadSignals(true)}
            disabled={isLoading}
          >
            {isLoading ? 'Refreshing...' : 'Refresh'}
          </Button>
        </div>
      </div>

      {/* Market Regime */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center justify-between">
            <span>Current Market Regime</span>
            <Badge 
              variant="outline" 
              className={getRegimeColor(signals.volatilityRegime)}
            >
              {signals.volatilityRegime.replace(/_/g, ' ').toUpperCase()}
            </Badge>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-muted-foreground">
            {getRegimeDescription(signals.volatilityRegime)}
          </p>
          {signals.semanticSearch?.interpretation && (
            <div className="mt-4 p-4 rounded-lg bg-blue-500/10 border border-blue-500/20">
              <p className="text-sm">{signals.semanticSearch.interpretation}</p>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Main Signals Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Momentum Signals */}
        <Card>
          <CardHeader>
            <CardTitle>
              Momentum Signals 
              <span className="text-sm font-normal text-muted-foreground ml-2">
                (12M return, skip 1M)
              </span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            {signals.momentum.length > 0 ? (
              <div className="space-y-2">
                {signals.momentum.map((s) => (
                  <div key={s.symbol} className="flex items-center justify-between">
                    <span className="font-mono font-medium">{s.symbol}</span>
                    <div className="flex items-center gap-2">
                      <div className="w-32 h-2 bg-muted rounded overflow-hidden">
                        <div 
                          className={cn(
                            'h-full transition-all',
                            s.score >= 0 ? 'bg-green-500' : 'bg-red-500'
                          )}
                          style={{ 
                            width: `${Math.abs(s.score) * 100}%`,
                            marginLeft: s.score < 0 ? 'auto' : 0 
                          }}
                        />
                      </div>
                      <span className={cn(
                        'font-mono text-sm w-16 text-right',
                        s.score >= 0 ? 'text-green-500' : 'text-red-500'
                      )}>
                        {s.score >= 0 ? '+' : ''}{(s.score * 100).toFixed(0)}%
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-muted-foreground text-center py-4">
                No momentum signals available
              </p>
            )}
          </CardContent>
        </Card>

        {/* Mean Reversion Signals */}
        <Card>
          <CardHeader>
            <CardTitle>
              Mean Reversion Signals
              <span className="text-sm font-normal text-muted-foreground ml-2">
                (Bollinger Z-Score)
              </span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            {signals.meanReversion.length > 0 ? (
              <div className="space-y-2">
                {signals.meanReversion.map((s) => (
                  <div key={s.symbol} className="flex items-center justify-between">
                    <span className="font-mono font-medium">{s.symbol}</span>
                    <div className="flex items-center gap-2">
                      <div className="w-32 h-2 bg-muted rounded overflow-hidden">
                        <div 
                          className={cn(
                            'h-full transition-all',
                            s.score >= 0 ? 'bg-green-500' : 'bg-red-500'
                          )}
                          style={{ 
                            width: `${Math.abs(s.score) * 100}%`,
                            marginLeft: s.score < 0 ? 'auto' : 0 
                          }}
                        />
                      </div>
                      <span className={cn(
                        'font-mono text-sm w-16 text-right',
                        s.score >= 0 ? 'text-green-500' : 'text-red-500'
                      )}>
                        {s.score >= 0 ? '+' : ''}{(s.score * 100).toFixed(0)}%
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-muted-foreground text-center py-4">
                No mean reversion signals available
              </p>
            )}
            <p className="text-xs text-muted-foreground mt-4">
              Positive = oversold (buy signal), Negative = overbought (sell signal)
            </p>
          </CardContent>
        </Card>

        {/* Technical Indicators */}
        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle>Technical Indicators</CardTitle>
          </CardHeader>
          <CardContent>
            {signals.technical.length > 0 ? (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {signals.technical.map((t) => (
                  <div key={t.symbol} className="p-4 rounded-lg bg-muted/50">
                    <p className="font-mono font-bold text-lg mb-3">{t.symbol}</p>
                    <div className="grid grid-cols-3 gap-3 text-sm">
                      <div>
                        <span className="text-muted-foreground">RSI:</span>{' '}
                        <span className={cn(
                          'font-medium',
                          t.rsi > 70 ? 'text-red-500' : t.rsi < 30 ? 'text-green-500' : ''
                        )}>
                          {t.rsi.toFixed(0)}
                        </span>
                      </div>
                      <div>
                        <span className="text-muted-foreground">MACD:</span>{' '}
                        <span className={cn(
                          'font-medium',
                          t.macd.histogram > 0 ? 'text-green-500' : 'text-red-500'
                        )}>
                          {t.macd.histogram.toFixed(2)}
                        </span>
                      </div>
                      <div>
                        <span className="text-muted-foreground">ATR:</span>{' '}
                        <span className="font-medium">{t.atr.toFixed(2)}</span>
                      </div>
                    </div>
                    <div className="mt-3 pt-3 border-t border-border/50">
                      <div className="flex justify-between text-xs text-muted-foreground">
                        <span>SMA20: ${t.sma20.toFixed(0)}</span>
                        <span>SMA50: ${t.sma50.toFixed(0)}</span>
                        <span>SMA200: ${t.sma200.toFixed(0)}</span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-muted-foreground text-center py-4">
                No technical indicators available
              </p>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Semantic Search Section (if we have data) */}
      {signals.semanticSearch?.similarPeriods?.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Semantic Market Memory</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-3 gap-4">
              <div>
                <p className="text-sm text-muted-foreground">Avg 5-Day Return</p>
                <p className={cn(
                  'text-2xl font-mono font-bold',
                  signals.semanticSearch.avg5dReturn >= 0 ? 'text-green-500' : 'text-red-500'
                )}>
                  {signals.semanticSearch.avg5dReturn >= 0 ? '+' : ''}
                  {(signals.semanticSearch.avg5dReturn * 100).toFixed(2)}%
                </p>
              </div>
              <div>
                <p className="text-sm text-muted-foreground">Avg 20-Day Return</p>
                <p className={cn(
                  'text-2xl font-mono font-bold',
                  signals.semanticSearch.avg20dReturn >= 0 ? 'text-green-500' : 'text-red-500'
                )}>
                  {signals.semanticSearch.avg20dReturn >= 0 ? '+' : ''}
                  {(signals.semanticSearch.avg20dReturn * 100).toFixed(2)}%
                </p>
              </div>
              <div>
                <p className="text-sm text-muted-foreground">Positive 5D Rate</p>
                <p className="text-2xl font-mono font-bold">
                  {(signals.semanticSearch.positive5dRate * 100).toFixed(0)}%
                </p>
              </div>
            </div>
            
            <div>
              <p className="text-sm text-muted-foreground mb-2">Similar Historical Periods</p>
              <div className="space-y-2">
                {signals.semanticSearch.similarPeriods.map((p, i) => (
                  <div 
                    key={i}
                    className="flex items-center justify-between p-2 rounded bg-muted/50"
                  >
                    <span className="font-mono text-sm">{p.date}</span>
                    <Badge variant="outline">
                      {(p.similarity * 100).toFixed(0)}% match
                    </Badge>
                    <span className={cn(
                      'font-mono text-sm',
                      p.return5d >= 0 ? 'text-green-500' : 'text-red-500'
                    )}>
                      5D: {p.return5d >= 0 ? '+' : ''}{(p.return5d * 100).toFixed(1)}%
                    </span>
                    <span className={cn(
                      'font-mono text-sm',
                      p.return20d >= 0 ? 'text-green-500' : 'text-red-500'
                    )}>
                      20D: {p.return20d >= 0 ? '+' : ''}{(p.return20d * 100).toFixed(1)}%
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Footer with timestamp */}
      <div className="text-center text-xs text-muted-foreground">
        Last updated: {lastRefresh?.toLocaleString() || 'Never'} | 
        Data source: yfinance (Yahoo Finance)
      </div>
    </div>
  );
}
