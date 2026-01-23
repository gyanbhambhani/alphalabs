'use client';

import { useEffect, useState } from 'react';
import { Leaderboard } from '@/components/Leaderboard';
import { ManagerCard } from '@/components/ManagerCard';
import { SignalDisplay } from '@/components/SignalDisplay';
import { TradeHistory } from '@/components/TradeHistory';
import { PortfolioChart } from '@/components/PortfolioChart';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import type { 
  LeaderboardEntry, 
  StrategySignals, 
  Trade, 
  DailySnapshot 
} from '@/types';

// Mock data for development (will be replaced with API calls)
const MOCK_MANAGERS = [
  { id: 'gpt4', name: 'GPT-4 Fund', type: 'llm' as const, provider: 'openai' as const, isActive: true },
  { id: 'claude', name: 'Claude Fund', type: 'llm' as const, provider: 'anthropic' as const, isActive: true },
  { id: 'gemini', name: 'Gemini Fund', type: 'llm' as const, provider: 'google' as const, isActive: true },
  { id: 'quant', name: 'Quant Bot', type: 'quant' as const, provider: null, isActive: true },
];

const MOCK_LEADERBOARD: LeaderboardEntry[] = [
  {
    rank: 1,
    manager: MOCK_MANAGERS[1],
    portfolio: { managerId: 'claude', cashBalance: 12000, totalValue: 28400, updatedAt: new Date().toISOString() },
    sharpeRatio: 1.82,
    totalReturn: 0.124,
    volatility: 0.082,
    maxDrawdown: -0.042,
    totalTrades: 47,
    winRate: 0.62,
  },
  {
    rank: 2,
    manager: MOCK_MANAGERS[0],
    portfolio: { managerId: 'gpt4', cashBalance: 8000, totalValue: 28775, updatedAt: new Date().toISOString() },
    sharpeRatio: 1.54,
    totalReturn: 0.151,
    volatility: 0.118,
    maxDrawdown: -0.061,
    totalTrades: 62,
    winRate: 0.58,
  },
  {
    rank: 3,
    manager: MOCK_MANAGERS[3],
    portfolio: { managerId: 'quant', cashBalance: 15000, totalValue: 27050, updatedAt: new Date().toISOString() },
    sharpeRatio: 1.21,
    totalReturn: 0.082,
    volatility: 0.071,
    maxDrawdown: -0.038,
    totalTrades: 31,
    winRate: 0.65,
  },
  {
    rank: 4,
    manager: MOCK_MANAGERS[2],
    portfolio: { managerId: 'gemini', cashBalance: 5000, totalValue: 27750, updatedAt: new Date().toISOString() },
    sharpeRatio: 0.92,
    totalReturn: 0.11,
    volatility: 0.142,
    maxDrawdown: -0.085,
    totalTrades: 58,
    winRate: 0.52,
  },
];

const MOCK_SIGNALS: StrategySignals = {
  momentum: [
    { symbol: 'NVDA', score: 0.85 },
    { symbol: 'MSFT', score: 0.62 },
    { symbol: 'AAPL', score: 0.45 },
    { symbol: 'GOOGL', score: 0.38 },
    { symbol: 'META', score: 0.22 },
    { symbol: 'TSLA', score: -0.15 },
    { symbol: 'AMD', score: -0.32 },
  ],
  meanReversion: [
    { symbol: 'TSLA', score: 0.72 },
    { symbol: 'AMD', score: 0.58 },
    { symbol: 'NVDA', score: -0.45 },
    { symbol: 'MSFT', score: -0.22 },
  ],
  technical: [],
  mlPrediction: [
    { symbol: 'NVDA', predictedReturn: 0.023, confidence: 0.72 },
    { symbol: 'MSFT', predictedReturn: 0.015, confidence: 0.68 },
    { symbol: 'TSLA', predictedReturn: -0.018, confidence: 0.61 },
  ],
  volatilityRegime: 'low_vol_trending_up',
  semanticSearch: {
    similarPeriods: [
      { date: '2023-11-15', similarity: 0.92, return5d: 0.032, return20d: 0.078 },
      { date: '2021-03-22', similarity: 0.88, return5d: 0.025, return20d: 0.065 },
      { date: '2019-10-28', similarity: 0.85, return5d: 0.018, return20d: 0.042 },
      { date: '2024-02-12', similarity: 0.82, return5d: 0.028, return20d: 0.058 },
      { date: '2020-06-08', similarity: 0.79, return5d: -0.012, return20d: 0.035 },
    ],
    avg5dReturn: 0.0182,
    avg20dReturn: 0.0556,
    positive5dRate: 0.72,
    interpretation: 'Current market conditions resemble low-volatility tech rallies. ' +
      'Historically, similar periods led to continued gains with 72% positive outcomes over 5 days.',
  },
  timestamp: new Date().toISOString(),
};

const MOCK_TRADES: Trade[] = [
  {
    id: 1,
    managerId: 'gpt4',
    symbol: 'NVDA',
    side: 'buy',
    quantity: 15,
    price: 142.50,
    reasoning: 'Strong momentum signal (+0.85) aligned with semantic search showing 72% positive outcomes in similar periods.',
    executedAt: new Date(Date.now() - 1000 * 60 * 30).toISOString(),
  },
  {
    id: 2,
    managerId: 'claude',
    symbol: 'MSFT',
    side: 'buy',
    quantity: 20,
    price: 425.30,
    reasoning: 'Low volatility trending regime favors quality tech. ML prediction shows +1.5% expected return.',
    executedAt: new Date(Date.now() - 1000 * 60 * 60).toISOString(),
  },
  {
    id: 3,
    managerId: 'quant',
    symbol: 'NVDA',
    side: 'buy',
    quantity: 10,
    price: 141.80,
    signalsUsed: { momentum: 0.85, semantic: 0.72 },
    executedAt: new Date(Date.now() - 1000 * 60 * 90).toISOString(),
  },
];

export default function Dashboard() {
  const [leaderboard, setLeaderboard] = useState<LeaderboardEntry[]>([]);
  const [signals, setSignals] = useState<StrategySignals | null>(null);
  const [trades, setTrades] = useState<Trade[]>([]);
  const [snapshots] = useState<Record<string, DailySnapshot[]>>({});
  const [isLive, setIsLive] = useState(false);

  useEffect(() => {
    // Load mock data
    setLeaderboard(MOCK_LEADERBOARD);
    setSignals(MOCK_SIGNALS);
    setTrades(MOCK_TRADES);
    
    // Check if API is available
    fetch('/api/health')
      .then(() => setIsLive(true))
      .catch(() => setIsLive(false));
  }, []);

  const managerNames: Record<string, string> = {};
  MOCK_MANAGERS.forEach((m) => {
    managerNames[m.id] = m.name;
  });

  const totalValue = leaderboard.reduce(
    (sum, e) => sum + e.portfolio.totalValue,
    0
  );
  
  const avgReturn = leaderboard.length > 0
    ? leaderboard.reduce((sum, e) => sum + e.totalReturn, 0) / leaderboard.length
    : 0;

  return (
    <div className="space-y-6">
      {/* Header Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm text-muted-foreground">
              Total AUM
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-mono font-bold">
              ${totalValue.toLocaleString()}
            </p>
          </CardContent>
        </Card>
        
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm text-muted-foreground">
              Avg Return
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className={`text-2xl font-mono font-bold ${avgReturn >= 0 ? 'text-green-500' : 'text-red-500'}`}>
              {avgReturn >= 0 ? '+' : ''}{(avgReturn * 100).toFixed(2)}%
            </p>
          </CardContent>
        </Card>
        
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm text-muted-foreground">
              Active Managers
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-mono font-bold">
              {leaderboard.length}
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
              className={isLive 
                ? 'bg-green-500/20 text-green-500' 
                : 'bg-yellow-500/20 text-yellow-500'
              }
            >
              {isLive ? '● Live' : '○ Demo Mode'}
            </Badge>
          </CardContent>
        </Card>
      </div>

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Leaderboard - Full width on mobile, 2 cols on desktop */}
        <div className="lg:col-span-2">
          <Leaderboard 
            entries={leaderboard}
            onSelectManager={(id) => {
              window.location.href = `/managers/${id}`;
            }}
          />
        </div>
        
        {/* Signals Panel */}
        <div>
          <SignalDisplay signals={signals} />
        </div>
      </div>

      {/* Performance Chart */}
      <PortfolioChart snapshots={snapshots} managerNames={managerNames} />

      {/* Manager Cards */}
      <div>
        <h2 className="text-lg font-semibold mb-4">Portfolio Managers</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {leaderboard.map((entry) => (
            <ManagerCard
              key={entry.manager.id}
              manager={entry.manager}
              portfolio={entry.portfolio}
              sharpeRatio={entry.sharpeRatio}
              totalReturn={entry.totalReturn}
              onClick={() => {
                window.location.href = `/managers/${entry.manager.id}`;
              }}
            />
          ))}
        </div>
      </div>

      {/* Recent Trades */}
      <TradeHistory trades={trades} managerNames={managerNames} />
    </div>
  );
}
