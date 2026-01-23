'use client';

import { useEffect, useState } from 'react';
import { SignalDisplay } from '@/components/SignalDisplay';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import type { StrategySignals } from '@/types';
import { cn } from '@/lib/utils';

const MOCK_SIGNALS: StrategySignals = {
  momentum: [
    { symbol: 'NVDA', score: 0.85 },
    { symbol: 'MSFT', score: 0.62 },
    { symbol: 'AAPL', score: 0.45 },
    { symbol: 'GOOGL', score: 0.38 },
    { symbol: 'META', score: 0.22 },
    { symbol: 'AMZN', score: 0.18 },
    { symbol: 'TSLA', score: -0.15 },
    { symbol: 'AMD', score: -0.32 },
    { symbol: 'INTC', score: -0.45 },
  ],
  meanReversion: [
    { symbol: 'TSLA', score: 0.72 },
    { symbol: 'AMD', score: 0.58 },
    { symbol: 'INTC', score: 0.45 },
    { symbol: 'META', score: 0.12 },
    { symbol: 'NVDA', score: -0.45 },
    { symbol: 'MSFT', score: -0.22 },
    { symbol: 'AAPL', score: -0.18 },
  ],
  technical: [
    { symbol: 'NVDA', rsi: 68, macd: { macd: 2.5, signal: 2.1, histogram: 0.4 }, sma20: 138, sma50: 132, sma200: 115, atr: 4.2 },
    { symbol: 'MSFT', rsi: 62, macd: { macd: 1.8, signal: 1.5, histogram: 0.3 }, sma20: 420, sma50: 412, sma200: 385, atr: 6.5 },
    { symbol: 'AAPL', rsi: 55, macd: { macd: 0.9, signal: 0.7, histogram: 0.2 }, sma20: 178, sma50: 175, sma200: 168, atr: 2.8 },
  ],
  mlPrediction: [
    { symbol: 'NVDA', predictedReturn: 0.023, confidence: 0.72 },
    { symbol: 'MSFT', predictedReturn: 0.015, confidence: 0.68 },
    { symbol: 'META', predictedReturn: 0.018, confidence: 0.65 },
    { symbol: 'GOOGL', predictedReturn: 0.012, confidence: 0.62 },
    { symbol: 'AAPL', predictedReturn: 0.008, confidence: 0.58 },
    { symbol: 'TSLA', predictedReturn: -0.018, confidence: 0.61 },
    { symbol: 'AMD', predictedReturn: -0.012, confidence: 0.55 },
  ],
  volatilityRegime: 'low_vol_trending_up',
  semanticSearch: {
    similarPeriods: [
      { date: '2023-11-15', similarity: 0.92, return5d: 0.032, return20d: 0.078 },
      { date: '2021-03-22', similarity: 0.88, return5d: 0.025, return20d: 0.065 },
      { date: '2019-10-28', similarity: 0.85, return5d: 0.018, return20d: 0.042 },
      { date: '2024-02-12', similarity: 0.82, return5d: 0.028, return20d: 0.058 },
      { date: '2020-06-08', similarity: 0.79, return5d: -0.012, return20d: 0.035 },
      { date: '2017-09-18', similarity: 0.76, return5d: 0.015, return20d: 0.028 },
      { date: '2018-01-22', similarity: 0.74, return5d: 0.022, return20d: -0.045 },
    ],
    avg5dReturn: 0.0182,
    avg20dReturn: 0.0556,
    positive5dRate: 0.72,
    interpretation: 'Current market conditions resemble low-volatility tech rallies from the past decade. Historically, similar periods led to continued gains with 72% positive outcomes over 5 days and an average 20-day return of +5.6%.',
  },
  timestamp: new Date().toISOString(),
};

export default function SignalsPage() {
  const [signals, setSignals] = useState<StrategySignals | null>(null);

  useEffect(() => {
    setSignals(MOCK_SIGNALS);
  }, []);

  if (!signals) {
    return (
      <div className="flex items-center justify-center h-64">
        <p className="text-muted-foreground">Loading signals...</p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold mb-2">Strategy Signals</h1>
        <p className="text-muted-foreground">
          Real-time signals from the shared strategy toolbox
        </p>
      </div>

      {/* Market Regime */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center justify-between">
            <span>Current Market Regime</span>
            <Badge 
              variant="outline" 
              className={cn(
                signals.volatilityRegime.includes('trending_up') 
                  ? 'bg-green-500/20 text-green-500'
                  : signals.volatilityRegime.includes('trending_down')
                  ? 'bg-red-500/20 text-red-500'
                  : 'bg-yellow-500/20 text-yellow-500'
              )}
            >
              {signals.volatilityRegime.replace(/_/g, ' ').toUpperCase()}
            </Badge>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-muted-foreground">
            {signals.volatilityRegime.includes('low_vol') && 'Low volatility environment - favorable for momentum strategies.'}
            {signals.volatilityRegime.includes('normal_vol') && 'Normal volatility - balanced approach recommended.'}
            {signals.volatilityRegime.includes('high_vol') && 'High volatility - caution advised, mean reversion may work better.'}
          </p>
        </CardContent>
      </Card>

      {/* Main Signals */}
      <SignalDisplay signals={signals} />

      {/* Detailed Signal Tables */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Momentum Details */}
        <Card>
          <CardHeader>
            <CardTitle>Momentum Signals (12M return, skip 1M)</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              {signals.momentum.map((s) => (
                <div key={s.symbol} className="flex items-center justify-between">
                  <span className="font-mono">{s.symbol}</span>
                  <div className="flex items-center gap-2">
                    <div className="w-32 h-2 bg-muted rounded overflow-hidden">
                      <div 
                        className={cn(
                          'h-full',
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
          </CardContent>
        </Card>

        {/* Mean Reversion Details */}
        <Card>
          <CardHeader>
            <CardTitle>Mean Reversion Signals (Bollinger Z-Score)</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              {signals.meanReversion.map((s) => (
                <div key={s.symbol} className="flex items-center justify-between">
                  <span className="font-mono">{s.symbol}</span>
                  <div className="flex items-center gap-2">
                    <div className="w-32 h-2 bg-muted rounded overflow-hidden">
                      <div 
                        className={cn(
                          'h-full',
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
            <p className="text-xs text-muted-foreground mt-4">
              Positive = oversold (buy signal), Negative = overbought (sell signal)
            </p>
          </CardContent>
        </Card>

        {/* ML Predictions */}
        <Card>
          <CardHeader>
            <CardTitle>ML Predictions (5-Day Forward Return)</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              {signals.mlPrediction.map((s) => (
                <div key={s.symbol} className="flex items-center justify-between">
                  <span className="font-mono">{s.symbol}</span>
                  <div className="flex items-center gap-4">
                    <span className={cn(
                      'font-mono text-sm',
                      s.predictedReturn >= 0 ? 'text-green-500' : 'text-red-500'
                    )}>
                      {s.predictedReturn >= 0 ? '+' : ''}{(s.predictedReturn * 100).toFixed(2)}%
                    </span>
                    <Badge variant="outline" className="text-xs">
                      {(s.confidence * 100).toFixed(0)}% conf
                    </Badge>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Technical Indicators */}
        <Card>
          <CardHeader>
            <CardTitle>Technical Indicators</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {signals.technical.slice(0, 5).map((t) => (
                <div key={t.symbol} className="p-3 rounded-lg bg-muted/50">
                  <p className="font-mono font-bold mb-2">{t.symbol}</p>
                  <div className="grid grid-cols-3 gap-2 text-xs">
                    <div>
                      <span className="text-muted-foreground">RSI:</span>{' '}
                      <span className={cn(
                        t.rsi > 70 ? 'text-red-500' : t.rsi < 30 ? 'text-green-500' : ''
                      )}>
                        {t.rsi}
                      </span>
                    </div>
                    <div>
                      <span className="text-muted-foreground">MACD:</span>{' '}
                      <span className={cn(
                        t.macd.histogram > 0 ? 'text-green-500' : 'text-red-500'
                      )}>
                        {t.macd.histogram.toFixed(2)}
                      </span>
                    </div>
                    <div>
                      <span className="text-muted-foreground">ATR:</span>{' '}
                      {t.atr.toFixed(2)}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Semantic Search Deep Dive */}
      <Card>
        <CardHeader>
          <CardTitle>Semantic Market Memory</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="p-4 rounded-lg bg-blue-500/10 border border-blue-500/20">
            <p className="text-sm">{signals.semanticSearch.interpretation}</p>
          </div>
          
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
    </div>
  );
}
