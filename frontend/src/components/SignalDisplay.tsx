'use client';

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import type { StrategySignals } from '@/types';
import { cn } from '@/lib/utils';

interface SignalDisplayProps {
  signals: StrategySignals | null;
}

function SignalBar({ value, label }: { value: number; label: string }) {
  const isPositive = value >= 0;
  const width = Math.min(Math.abs(value) * 100, 100);
  
  return (
    <div className="flex items-center gap-2">
      <span className="w-16 text-xs font-mono">{label}</span>
      <div className="flex-1 h-4 bg-muted rounded relative overflow-hidden">
        <div className="absolute inset-0 flex items-center justify-center">
          <div
            className={cn(
              'h-full absolute',
              isPositive ? 'bg-green-500/50 left-1/2' : 'bg-red-500/50 right-1/2'
            )}
            style={{ width: `${width / 2}%` }}
          />
        </div>
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="w-px h-full bg-border" />
        </div>
      </div>
      <span
        className={cn(
          'w-14 text-xs font-mono text-right',
          isPositive ? 'text-green-500' : 'text-red-500'
        )}
      >
        {value >= 0 ? '+' : ''}{(value * 100).toFixed(0)}%
      </span>
    </div>
  );
}

function getRegimeBadgeColor(regime: string) {
  if (regime.includes('trending_up')) return 'bg-green-500/20 text-green-500';
  if (regime.includes('trending_down')) return 'bg-red-500/20 text-red-500';
  return 'bg-yellow-500/20 text-yellow-500';
}

export function SignalDisplay({ signals }: SignalDisplayProps) {
  if (!signals) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Strategy Signals</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-muted-foreground">
            Waiting for market data...
          </p>
        </CardContent>
      </Card>
    );
  }
  
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          <span>Strategy Signals</span>
          <Badge 
            variant="outline" 
            className={getRegimeBadgeColor(signals.volatilityRegime)}
          >
            {signals.volatilityRegime.replace(/_/g, ' ')}
          </Badge>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <Tabs defaultValue="momentum">
          <TabsList className="grid w-full grid-cols-4">
            <TabsTrigger value="momentum">Momentum</TabsTrigger>
            <TabsTrigger value="meanrev">Mean Rev.</TabsTrigger>
            <TabsTrigger value="ml">ML Pred.</TabsTrigger>
            <TabsTrigger value="semantic">Semantic</TabsTrigger>
          </TabsList>
          
          <TabsContent value="momentum" className="space-y-2 mt-4">
            {signals.momentum.length === 0 ? (
              <p className="text-sm text-muted-foreground">No momentum signals</p>
            ) : (
              signals.momentum.slice(0, 10).map((s) => (
                <SignalBar key={s.symbol} value={s.score} label={s.symbol} />
              ))
            )}
          </TabsContent>
          
          <TabsContent value="meanrev" className="space-y-2 mt-4">
            {signals.meanReversion.length === 0 ? (
              <p className="text-sm text-muted-foreground">
                No mean reversion signals
              </p>
            ) : (
              signals.meanReversion.slice(0, 10).map((s) => (
                <SignalBar key={s.symbol} value={s.score} label={s.symbol} />
              ))
            )}
          </TabsContent>
          
          <TabsContent value="ml" className="space-y-2 mt-4">
            {signals.mlPrediction.length === 0 ? (
              <p className="text-sm text-muted-foreground">No ML predictions</p>
            ) : (
              signals.mlPrediction.slice(0, 10).map((s) => (
                <SignalBar 
                  key={s.symbol} 
                  value={s.predictedReturn} 
                  label={s.symbol} 
                />
              ))
            )}
          </TabsContent>
          
          <TabsContent value="semantic" className="mt-4">
            <div className="space-y-4">
              <div className="grid grid-cols-3 gap-4">
                <div>
                  <p className="text-xs text-muted-foreground">Avg 5D Return</p>
                  <p
                    className={cn(
                      'text-lg font-mono',
                      signals.semanticSearch.avg5dReturn >= 0
                        ? 'text-green-500'
                        : 'text-red-500'
                    )}
                  >
                    {signals.semanticSearch.avg5dReturn >= 0 ? '+' : ''}
                    {(signals.semanticSearch.avg5dReturn * 100).toFixed(2)}%
                  </p>
                </div>
                <div>
                  <p className="text-xs text-muted-foreground">Avg 20D Return</p>
                  <p
                    className={cn(
                      'text-lg font-mono',
                      signals.semanticSearch.avg20dReturn >= 0
                        ? 'text-green-500'
                        : 'text-red-500'
                    )}
                  >
                    {signals.semanticSearch.avg20dReturn >= 0 ? '+' : ''}
                    {(signals.semanticSearch.avg20dReturn * 100).toFixed(2)}%
                  </p>
                </div>
                <div>
                  <p className="text-xs text-muted-foreground">Positive Rate</p>
                  <p className="text-lg font-mono">
                    {(signals.semanticSearch.positive5dRate * 100).toFixed(0)}%
                  </p>
                </div>
              </div>
              
              <div className="p-3 bg-muted rounded-lg">
                <p className="text-sm font-medium mb-1">Market Memory</p>
                <p className="text-xs text-muted-foreground">
                  {signals.semanticSearch.interpretation || 
                    `Found ${signals.semanticSearch.similarPeriods.length} similar ` +
                    `historical periods.`}
                </p>
              </div>
              
              {signals.semanticSearch.similarPeriods.length > 0 && (
                <div>
                  <p className="text-xs text-muted-foreground mb-2">
                    Top Similar Periods
                  </p>
                  <div className="space-y-1">
                    {signals.semanticSearch.similarPeriods.slice(0, 5).map((p, i) => (
                      <div
                        key={i}
                        className="flex justify-between text-xs font-mono"
                      >
                        <span>{p.date}</span>
                        <span className="text-muted-foreground">
                          {(p.similarity * 100).toFixed(0)}% match
                        </span>
                        <span
                          className={cn(
                            p.return5d >= 0 ? 'text-green-500' : 'text-red-500'
                          )}
                        >
                          {p.return5d >= 0 ? '+' : ''}
                          {(p.return5d * 100).toFixed(1)}%
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
}
