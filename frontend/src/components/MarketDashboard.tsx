'use client';

import { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Separator } from '@/components/ui/separator';
import {
  TrendingUp,
  TrendingDown,
  Activity,
  AlertTriangle,
  Target,
  Search,
  Clock,
} from 'lucide-react';

interface MarketContext {
  symbol: string;
  regime: string;
  volatility: number;
  momentum_1m: number;
  momentum_3m?: number;
  recommendation: string;
  confidence: number;
  interpretation: string;
  avg_forward_return_1m?: number;
  positive_outcome_rate?: number;
  worst_case_drawdown?: number;
  key_risks?: string[];
  similar_periods: Array<{
    date: string;
    similarity: number;
    regime: string;
    narrative: string;
    forward_return_1m?: number;
  }>;
}

interface MarketSentiment {
  sentiment: {
    level: string;
    vix: number;
    vix_percentile: number;
    breadth?: number;
    interpretation: string;
  };
  economic: {
    ten_year_yield: number;
    yield_curve_spread: number;
    recession_risk: boolean;
    interpretation?: string;
  };
  narrative: string;
  geopolitical?: string;
}

interface MarketDashboardProps {
  symbol: string;
  context: MarketContext | null;
  sentiment: MarketSentiment | null;
  onSymbolChangeAction: (symbol: string) => void;
}

export function MarketDashboard({
  symbol,
  context,
  sentiment,
  onSymbolChangeAction,
}: MarketDashboardProps) {
  const [symbolInput, setSymbolInput] = useState(symbol);

  const handleSymbolSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (symbolInput.trim()) {
      onSymbolChangeAction(symbolInput.toUpperCase().trim());
    }
  };

  const getRegimeColor = (regime?: string) => {
    if (!regime) return 'bg-blue-500/20 text-blue-500';
    if (regime.includes('bull') || regime.includes('recovery')) {
      return 'bg-green-500/20 text-green-500';
    }
    if (regime.includes('bear') || regime.includes('capitulation')) {
      return 'bg-red-500/20 text-red-500';
    }
    if (regime.includes('euphoria')) {
      return 'bg-yellow-500/20 text-yellow-500';
    }
    return 'bg-blue-500/20 text-blue-500';
  };

  const getRecommendationColor = (rec?: string) => {
    if (!rec) return 'text-yellow-500';
    if (rec.includes('long') || rec.includes('aggressive')) {
      return 'text-green-500';
    }
    if (rec.includes('defensive') || rec.includes('reduce')) {
      return 'text-red-500';
    }
    return 'text-yellow-500';
  };

  return (
    <div className="space-y-4">
      {/* Symbol Search */}
      <form onSubmit={handleSymbolSubmit} className="flex gap-2">
        <Input
          value={symbolInput}
          onChange={(e) => setSymbolInput(e.target.value.toUpperCase())}
          placeholder="Enter symbol (e.g., AAPL)"
          className="flex-1"
        />
        <Button type="submit" variant="outline">
          <Search className="h-4 w-4" />
        </Button>
      </form>

      {/* Market Sentiment Card */}
      {sentiment && (
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium flex items-center gap-2">
              <Activity className="h-4 w-4" />
              Market Sentiment
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <p className="text-xs text-muted-foreground">VIX Level</p>
                <p className="text-xl font-bold">
                  {sentiment.sentiment.vix.toFixed(1)}
                </p>
                <p className="text-xs text-muted-foreground">
                  {(sentiment.sentiment.vix_percentile * 100).toFixed(0)}% percentile
                </p>
              </div>
              <div>
                <p className="text-xs text-muted-foreground">Sentiment</p>
                <Badge
                  className={
                    sentiment.sentiment.level.includes('fear')
                      ? 'bg-red-500/20 text-red-500'
                      : sentiment.sentiment.level.includes('greed')
                      ? 'bg-green-500/20 text-green-500'
                      : 'bg-yellow-500/20 text-yellow-500'
                  }
                >
                  {sentiment.sentiment.level.replace('_', ' ')}
                </Badge>
              </div>
            </div>
            
            <Separator />
            
            <div className="grid grid-cols-2 gap-4">
              <div>
                <p className="text-xs text-muted-foreground">10Y Yield</p>
                <p className="font-medium">
                  {(sentiment.economic.ten_year_yield * 100).toFixed(2)}%
                </p>
              </div>
              <div>
                <p className="text-xs text-muted-foreground">Yield Curve</p>
                <p
                  className={`font-medium ${
                    sentiment.economic.yield_curve_spread < 0
                      ? 'text-red-500'
                      : 'text-green-500'
                  }`}
                >
                  {(sentiment.economic.yield_curve_spread * 100).toFixed(0)}bp
                </p>
              </div>
            </div>
            
            {sentiment.economic.recession_risk && (
              <div className="flex items-center gap-2 text-amber-500 text-sm">
                <AlertTriangle className="h-4 w-4" />
                Elevated recession risk
              </div>
            )}
            
            <p className="text-xs text-muted-foreground">
              {sentiment.narrative}
            </p>
          </CardContent>
        </Card>
      )}

      {/* Symbol Context Card */}
      {context && (
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium flex items-center gap-2">
              <Target className="h-4 w-4" />
              {context.symbol} Analysis
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex items-center justify-between">
              <Badge className={getRegimeColor(context.regime)}>
                {(context.regime || 'unknown').replace('_', ' ')}
              </Badge>
              <div className="text-right">
                <p className="text-xs text-muted-foreground">Confidence</p>
                <p className="font-bold">{(context.confidence * 100).toFixed(0)}%</p>
              </div>
            </div>
            
            <div className="grid grid-cols-3 gap-4">
              <div>
                <p className="text-xs text-muted-foreground">Volatility</p>
                <p className="font-medium">
                  {(context.volatility * 100).toFixed(1)}%
                </p>
              </div>
              <div>
                <p className="text-xs text-muted-foreground">Momentum 1M</p>
                <p
                  className={`font-medium ${
                    context.momentum_1m > 0 ? 'text-green-500' : 'text-red-500'
                  }`}
                >
                  {(context.momentum_1m * 100).toFixed(1)}%
                </p>
              </div>
              <div>
                <p className="text-xs text-muted-foreground">Avg Fwd Return</p>
                <p
                  className={`font-medium ${
                    (context.avg_forward_return_1m || 0) > 0
                      ? 'text-green-500'
                      : 'text-red-500'
                  }`}
                >
                  {((context.avg_forward_return_1m || 0) * 100).toFixed(1)}%
                </p>
              </div>
            </div>
            
            <Separator />
            
            <div>
              <p className="text-xs text-muted-foreground mb-1">Recommendation</p>
              <p
                className={`font-semibold ${getRecommendationColor(
                  context.recommendation
                )}`}
              >
                {(context.recommendation || 'neutral').replace('_', ' ').toUpperCase()}
              </p>
            </div>
            
            <p className="text-sm text-muted-foreground">
              {context.interpretation}
            </p>
            
            {/* Key Risks */}
            {context.key_risks && context.key_risks.length > 0 && (
              <div>
                <p className="text-xs text-muted-foreground mb-2 flex items-center gap-1">
                  <AlertTriangle className="h-3 w-3" />
                  Key Risks
                </p>
                <ul className="text-xs space-y-1">
                  {context.key_risks.slice(0, 3).map((risk, idx) => (
                    <li key={idx} className="text-muted-foreground">
                      • {risk}
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {/* Similar Historical Periods */}
      {context && context.similar_periods && context.similar_periods.length > 0 && (
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium flex items-center gap-2">
              <Clock className="h-4 w-4" />
              Similar Historical Periods
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {context.similar_periods.slice(0, 5).map((period, idx) => (
                <div
                  key={idx}
                  className="border-b border-border last:border-0 pb-3 last:pb-0"
                >
                  <div className="flex items-center justify-between mb-1">
                    <span className="font-medium text-sm">{period.date}</span>
                    <Badge variant="outline" className="text-xs">
                      {(period.similarity * 100).toFixed(0)}% match
                    </Badge>
                  </div>
                  <p className="text-xs text-muted-foreground line-clamp-2">
                    {period.narrative}
                  </p>
                  {period.forward_return_1m !== undefined && (
                    <p
                      className={`text-xs mt-1 ${
                        period.forward_return_1m > 0
                          ? 'text-green-500'
                          : 'text-red-500'
                      }`}
                    >
                      {period.forward_return_1m > 0 ? (
                        <TrendingUp className="h-3 w-3 inline mr-1" />
                      ) : (
                        <TrendingDown className="h-3 w-3 inline mr-1" />
                      )}
                      {(period.forward_return_1m * 100).toFixed(1)}% (1M outcome)
                    </p>
                  )}
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
