'use client';

import { useEffect, useState } from 'react';
import { useParams } from 'next/navigation';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Separator } from '@/components/ui/separator';
import { TradeHistory } from '@/components/TradeHistory';
import type { Manager, Portfolio, Position, Trade, DailySnapshot } from '@/types';
import { cn } from '@/lib/utils';

// Mock data
const MOCK_MANAGERS: Record<string, Manager> = {
  gpt4: { 
    id: 'gpt4', 
    name: 'GPT-4 Fund', 
    type: 'llm', 
    provider: 'openai', 
    isActive: true,
    description: 'Full reasoning with narrative understanding. Can read news context and make nuanced decisions.'
  },
  claude: { 
    id: 'claude', 
    name: 'Claude Fund', 
    type: 'llm', 
    provider: 'anthropic', 
    isActive: true,
    description: 'Deep reasoning with careful analysis. Good at handling uncertainty and edge cases.'
  },
  gemini: { 
    id: 'gemini', 
    name: 'Gemini Fund', 
    type: 'llm', 
    provider: 'google', 
    isActive: true,
    description: 'Fast multi-modal analysis. Can process charts and data simultaneously.'
  },
  quant: { 
    id: 'quant', 
    name: 'Quant Bot', 
    type: 'quant', 
    provider: null, 
    isActive: true,
    description: 'Pure systematic trading with fixed rules. Combines signals with predetermined weights. No reasoning, no interpretation.'
  },
};

const MOCK_PORTFOLIOS: Record<string, Portfolio> = {
  gpt4: { managerId: 'gpt4', cashBalance: 8000, totalValue: 28775, updatedAt: new Date().toISOString() },
  claude: { managerId: 'claude', cashBalance: 12000, totalValue: 28400, updatedAt: new Date().toISOString() },
  gemini: { managerId: 'gemini', cashBalance: 5000, totalValue: 27750, updatedAt: new Date().toISOString() },
  quant: { managerId: 'quant', cashBalance: 15000, totalValue: 27050, updatedAt: new Date().toISOString() },
};

const MOCK_POSITIONS: Record<string, Position[]> = {
  gpt4: [
    { id: 1, managerId: 'gpt4', symbol: 'NVDA', quantity: 50, avgEntryPrice: 135, currentPrice: 142.5, unrealizedPnl: 375, openedAt: new Date(Date.now() - 86400000 * 3).toISOString() },
    { id: 2, managerId: 'gpt4', symbol: 'MSFT', quantity: 30, avgEntryPrice: 420, currentPrice: 425, unrealizedPnl: 150, openedAt: new Date(Date.now() - 86400000 * 5).toISOString() },
    { id: 3, managerId: 'gpt4', symbol: 'AAPL', quantity: 25, avgEntryPrice: 180, currentPrice: 182, unrealizedPnl: 50, openedAt: new Date(Date.now() - 86400000 * 2).toISOString() },
  ],
  claude: [
    { id: 4, managerId: 'claude', symbol: 'AAPL', quantity: 40, avgEntryPrice: 178, currentPrice: 182, unrealizedPnl: 160, openedAt: new Date(Date.now() - 86400000 * 4).toISOString() },
    { id: 5, managerId: 'claude', symbol: 'GOOGL', quantity: 25, avgEntryPrice: 140, currentPrice: 145, unrealizedPnl: 125, openedAt: new Date(Date.now() - 86400000 * 6).toISOString() },
  ],
  gemini: [
    { id: 6, managerId: 'gemini', symbol: 'META', quantity: 35, avgEntryPrice: 480, currentPrice: 495, unrealizedPnl: 525, openedAt: new Date(Date.now() - 86400000 * 2).toISOString() },
    { id: 7, managerId: 'gemini', symbol: 'NVDA', quantity: 40, avgEntryPrice: 138, currentPrice: 142.5, unrealizedPnl: 180, openedAt: new Date(Date.now() - 86400000 * 1).toISOString() },
  ],
  quant: [
    { id: 8, managerId: 'quant', symbol: 'NVDA', quantity: 30, avgEntryPrice: 138, currentPrice: 142.5, unrealizedPnl: 135, openedAt: new Date(Date.now() - 86400000 * 4).toISOString() },
    { id: 9, managerId: 'quant', symbol: 'AAPL', quantity: 25, avgEntryPrice: 175, currentPrice: 182, unrealizedPnl: 175, openedAt: new Date(Date.now() - 86400000 * 7).toISOString() },
  ],
};

const MOCK_TRADES: Record<string, Trade[]> = {
  gpt4: [
    { id: 1, managerId: 'gpt4', symbol: 'NVDA', side: 'buy', quantity: 15, price: 142.50, reasoning: 'Strong momentum signal (+0.85) aligned with semantic search showing 72% positive outcomes.', executedAt: new Date(Date.now() - 1000 * 60 * 30).toISOString() },
    { id: 2, managerId: 'gpt4', symbol: 'AAPL', side: 'buy', quantity: 25, price: 180.00, reasoning: 'Diversifying into quality tech. Mean reversion signal suggests oversold conditions.', executedAt: new Date(Date.now() - 1000 * 60 * 60 * 24).toISOString() },
  ],
  claude: [
    { id: 3, managerId: 'claude', symbol: 'MSFT', side: 'buy', quantity: 20, price: 425.30, reasoning: 'Low volatility trending regime favors quality tech. ML prediction shows +1.5% expected return.', executedAt: new Date(Date.now() - 1000 * 60 * 60).toISOString() },
  ],
  gemini: [
    { id: 4, managerId: 'gemini', symbol: 'META', side: 'buy', quantity: 35, price: 480.00, reasoning: 'High conviction play on AI infrastructure buildout. Chart pattern suggests breakout.', executedAt: new Date(Date.now() - 1000 * 60 * 60 * 48).toISOString() },
  ],
  quant: [
    { id: 5, managerId: 'quant', symbol: 'NVDA', side: 'buy', quantity: 10, price: 141.80, signalsUsed: { momentum: 0.85, semantic: 0.72, mlPrediction: 0.023 }, executedAt: new Date(Date.now() - 1000 * 60 * 90).toISOString() },
  ],
};

const MOCK_PERFORMANCE: Record<string, { sharpe: number; return: number; volatility: number; maxDrawdown: number; winRate: number; totalTrades: number }> = {
  gpt4: { sharpe: 1.54, return: 0.151, volatility: 0.118, maxDrawdown: -0.061, winRate: 0.58, totalTrades: 62 },
  claude: { sharpe: 1.82, return: 0.124, volatility: 0.082, maxDrawdown: -0.042, winRate: 0.62, totalTrades: 47 },
  gemini: { sharpe: 0.92, return: 0.11, volatility: 0.142, maxDrawdown: -0.085, winRate: 0.52, totalTrades: 58 },
  quant: { sharpe: 1.21, return: 0.082, volatility: 0.071, maxDrawdown: -0.038, winRate: 0.65, totalTrades: 31 },
};

function formatCurrency(value: number): string {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
  }).format(value);
}

function formatPercent(value: number): string {
  const formatted = (value * 100).toFixed(2);
  return value >= 0 ? `+${formatted}%` : `${formatted}%`;
}

export default function ManagerDetailPage() {
  const params = useParams();
  const managerId = params.id as string;
  
  const [manager, setManager] = useState<Manager | null>(null);
  const [portfolio, setPortfolio] = useState<Portfolio | null>(null);
  const [positions, setPositions] = useState<Position[]>([]);
  const [trades, setTrades] = useState<Trade[]>([]);
  const [performance, setPerformance] = useState<typeof MOCK_PERFORMANCE.gpt4 | null>(null);

  useEffect(() => {
    if (managerId && MOCK_MANAGERS[managerId]) {
      setManager(MOCK_MANAGERS[managerId]);
      setPortfolio(MOCK_PORTFOLIOS[managerId]);
      setPositions(MOCK_POSITIONS[managerId] || []);
      setTrades(MOCK_TRADES[managerId] || []);
      setPerformance(MOCK_PERFORMANCE[managerId]);
    }
  }, [managerId]);

  if (!manager) {
    return (
      <div className="flex items-center justify-center h-64">
        <p className="text-muted-foreground">Loading...</p>
      </div>
    );
  }

  const isQuant = manager.type === 'quant';
  const totalPositionValue = positions.reduce(
    (sum, p) => sum + p.quantity * p.currentPrice,
    0
  );
  const totalUnrealizedPnl = positions.reduce(
    (sum, p) => sum + p.unrealizedPnl,
    0
  );

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div className="flex items-center gap-4">
          <span className="text-4xl">{isQuant ? '🤖' : '🧠'}</span>
          <div>
            <h1 className="text-2xl font-bold flex items-center gap-2">
              {manager.name}
              <Badge
                variant="outline"
                className={cn(
                  isQuant
                    ? 'bg-green-500/10 text-green-500'
                    : 'bg-blue-500/10 text-blue-500'
                )}
              >
                {isQuant ? 'Baseline' : manager.provider}
              </Badge>
            </h1>
            <p className="text-muted-foreground">{manager.description}</p>
          </div>
        </div>
        <Button variant="outline" onClick={() => window.history.back()}>
          Back
        </Button>
      </div>

      {/* Performance Metrics */}
      <div className="grid grid-cols-2 md:grid-cols-6 gap-4">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-xs text-muted-foreground">
              Sharpe Ratio
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-mono font-bold">
              {performance?.sharpe.toFixed(2) || '—'}
            </p>
          </CardContent>
        </Card>
        
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-xs text-muted-foreground">
              Total Return
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className={cn(
              'text-2xl font-mono font-bold',
              (performance?.return || 0) >= 0 ? 'text-green-500' : 'text-red-500'
            )}>
              {performance ? formatPercent(performance.return) : '—'}
            </p>
          </CardContent>
        </Card>
        
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-xs text-muted-foreground">
              Volatility
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-mono font-bold text-muted-foreground">
              {performance ? formatPercent(performance.volatility) : '—'}
            </p>
          </CardContent>
        </Card>
        
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-xs text-muted-foreground">
              Max Drawdown
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-mono font-bold text-red-500">
              {performance ? formatPercent(performance.maxDrawdown) : '—'}
            </p>
          </CardContent>
        </Card>
        
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-xs text-muted-foreground">
              Win Rate
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-mono font-bold">
              {performance ? `${(performance.winRate * 100).toFixed(0)}%` : '—'}
            </p>
          </CardContent>
        </Card>
        
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-xs text-muted-foreground">
              Total Trades
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-mono font-bold">
              {performance?.totalTrades || '—'}
            </p>
          </CardContent>
        </Card>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Portfolio Overview */}
        <Card>
          <CardHeader>
            <CardTitle>Portfolio Overview</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <p className="text-sm text-muted-foreground">Total Value</p>
                <p className="text-xl font-mono font-bold">
                  {portfolio ? formatCurrency(portfolio.totalValue) : '—'}
                </p>
              </div>
              <div>
                <p className="text-sm text-muted-foreground">Cash Balance</p>
                <p className="text-xl font-mono">
                  {portfolio ? formatCurrency(portfolio.cashBalance) : '—'}
                </p>
              </div>
              <div>
                <p className="text-sm text-muted-foreground">Invested</p>
                <p className="text-xl font-mono">
                  {formatCurrency(totalPositionValue)}
                </p>
              </div>
              <div>
                <p className="text-sm text-muted-foreground">Unrealized P&L</p>
                <p className={cn(
                  'text-xl font-mono',
                  totalUnrealizedPnl >= 0 ? 'text-green-500' : 'text-red-500'
                )}>
                  {formatCurrency(totalUnrealizedPnl)}
                </p>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Current Positions */}
        <Card>
          <CardHeader>
            <CardTitle>Current Positions ({positions.length})</CardTitle>
          </CardHeader>
          <CardContent>
            {positions.length === 0 ? (
              <p className="text-muted-foreground">No open positions</p>
            ) : (
              <div className="space-y-3">
                {positions.map((pos) => (
                  <div
                    key={pos.id}
                    className="flex items-center justify-between p-3 rounded-lg bg-muted/50"
                  >
                    <div>
                      <p className="font-mono font-bold">{pos.symbol}</p>
                      <p className="text-xs text-muted-foreground">
                        {pos.quantity} shares @ {formatCurrency(pos.avgEntryPrice)}
                      </p>
                    </div>
                    <div className="text-right">
                      <p className="font-mono">
                        {formatCurrency(pos.quantity * pos.currentPrice)}
                      </p>
                      <p className={cn(
                        'text-sm font-mono',
                        pos.unrealizedPnl >= 0 ? 'text-green-500' : 'text-red-500'
                      )}>
                        {pos.unrealizedPnl >= 0 ? '+' : ''}
                        {formatCurrency(pos.unrealizedPnl)}
                      </p>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Trade History */}
      <TradeHistory trades={trades} showManager={false} />

      {/* Manager-specific info */}
      {isQuant ? (
        <Card>
          <CardHeader>
            <CardTitle>Quant Bot Logic</CardTitle>
          </CardHeader>
          <CardContent>
            <pre className="p-4 rounded-lg bg-muted text-sm font-mono overflow-x-auto">
{`# Signal combination weights
score = (
    0.3 * momentum_signal +
    0.2 * mean_reversion_signal +
    0.2 * ml_prediction +
    0.3 * semantic_similarity_outcome
)

# Trading rules
if score > 0.6 and regime == "trending":
    BUY with size = f(score)
elif score < -0.6:
    SELL`}
            </pre>
          </CardContent>
        </Card>
      ) : (
        <Card>
          <CardHeader>
            <CardTitle>LLM Capabilities</CardTitle>
          </CardHeader>
          <CardContent>
            <ul className="space-y-2 text-sm">
              <li className="flex items-center gap-2">
                <span className="text-green-500">✓</span>
                Full autonomy to interpret strategy signals
              </li>
              <li className="flex items-center gap-2">
                <span className="text-green-500">✓</span>
                Can ignore signals if reasoning suggests otherwise
              </li>
              <li className="flex items-center gap-2">
                <span className="text-green-500">✓</span>
                Understands semantic market memory context
              </li>
              <li className="flex items-center gap-2">
                <span className="text-green-500">✓</span>
                Explains reasoning for every trade decision
              </li>
              <li className="flex items-center gap-2">
                <span className="text-green-500">✓</span>
                Adapts strategy based on market regime
              </li>
            </ul>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
