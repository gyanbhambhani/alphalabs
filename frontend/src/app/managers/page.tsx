'use client';

import { useEffect, useState } from 'react';
import { ManagerCard } from '@/components/ManagerCard';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import type { Manager, Portfolio, Position, Trade } from '@/types';

// Mock data
const MOCK_MANAGERS: Manager[] = [
  { id: 'gpt4', name: 'GPT-4 Fund', type: 'llm', provider: 'openai', isActive: true },
  { id: 'claude', name: 'Claude Fund', type: 'llm', provider: 'anthropic', isActive: true },
  { id: 'gemini', name: 'Gemini Fund', type: 'llm', provider: 'google', isActive: true },
  { id: 'quant', name: 'Quant Bot', type: 'quant', provider: null, isActive: true },
];

const MOCK_PORTFOLIOS: Record<string, Portfolio> = {
  gpt4: { managerId: 'gpt4', cashBalance: 8000, totalValue: 28775, updatedAt: new Date().toISOString() },
  claude: { managerId: 'claude', cashBalance: 12000, totalValue: 28400, updatedAt: new Date().toISOString() },
  gemini: { managerId: 'gemini', cashBalance: 5000, totalValue: 27750, updatedAt: new Date().toISOString() },
  quant: { managerId: 'quant', cashBalance: 15000, totalValue: 27050, updatedAt: new Date().toISOString() },
};

const MOCK_POSITIONS: Record<string, Position[]> = {
  gpt4: [
    { id: 1, managerId: 'gpt4', symbol: 'NVDA', quantity: 50, avgEntryPrice: 135, currentPrice: 142.5, unrealizedPnl: 375, openedAt: new Date().toISOString() },
    { id: 2, managerId: 'gpt4', symbol: 'MSFT', quantity: 30, avgEntryPrice: 420, currentPrice: 425, unrealizedPnl: 150, openedAt: new Date().toISOString() },
  ],
  claude: [
    { id: 3, managerId: 'claude', symbol: 'AAPL', quantity: 40, avgEntryPrice: 178, currentPrice: 182, unrealizedPnl: 160, openedAt: new Date().toISOString() },
    { id: 4, managerId: 'claude', symbol: 'GOOGL', quantity: 25, avgEntryPrice: 140, currentPrice: 145, unrealizedPnl: 125, openedAt: new Date().toISOString() },
  ],
  gemini: [
    { id: 5, managerId: 'gemini', symbol: 'META', quantity: 35, avgEntryPrice: 480, currentPrice: 495, unrealizedPnl: 525, openedAt: new Date().toISOString() },
  ],
  quant: [
    { id: 6, managerId: 'quant', symbol: 'NVDA', quantity: 30, avgEntryPrice: 138, currentPrice: 142.5, unrealizedPnl: 135, openedAt: new Date().toISOString() },
    { id: 7, managerId: 'quant', symbol: 'AAPL', quantity: 25, avgEntryPrice: 175, currentPrice: 182, unrealizedPnl: 175, openedAt: new Date().toISOString() },
  ],
};

const MOCK_PERFORMANCE: Record<string, { sharpe: number; return: number }> = {
  gpt4: { sharpe: 1.54, return: 0.151 },
  claude: { sharpe: 1.82, return: 0.124 },
  gemini: { sharpe: 0.92, return: 0.11 },
  quant: { sharpe: 1.21, return: 0.082 },
};

export default function ManagersPage() {
  const [managers, setManagers] = useState<Manager[]>([]);
  const [portfolios, setPortfolios] = useState<Record<string, Portfolio>>({});
  const [positions, setPositions] = useState<Record<string, Position[]>>({});

  useEffect(() => {
    setManagers(MOCK_MANAGERS);
    setPortfolios(MOCK_PORTFOLIOS);
    setPositions(MOCK_POSITIONS);
  }, []);

  const llmManagers = managers.filter((m) => m.type === 'llm');
  const quantManagers = managers.filter((m) => m.type === 'quant');

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-2xl font-bold mb-2">Portfolio Managers</h1>
        <p className="text-muted-foreground">
          4 autonomous AI portfolio managers competing on risk-adjusted returns
        </p>
      </div>

      {/* LLM Managers Section */}
      <div>
        <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
          <span>🧠</span>
          <span>LLM Portfolio Managers</span>
        </h2>
        <p className="text-sm text-muted-foreground mb-4">
          Full autonomy to interpret signals and make independent decisions
        </p>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {llmManagers.map((manager) => (
            <ManagerCard
              key={manager.id}
              manager={manager}
              portfolio={portfolios[manager.id]}
              positions={positions[manager.id] || []}
              sharpeRatio={MOCK_PERFORMANCE[manager.id]?.sharpe}
              totalReturn={MOCK_PERFORMANCE[manager.id]?.return}
              onClick={() => {
                window.location.href = `/managers/${manager.id}`;
              }}
            />
          ))}
        </div>
      </div>

      {/* Quant Bot Section */}
      <div>
        <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
          <span>🤖</span>
          <span>Quant Baseline</span>
        </h2>
        <p className="text-sm text-muted-foreground mb-4">
          Pure systematic trading with fixed rules - the benchmark to beat
        </p>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {quantManagers.map((manager) => (
            <ManagerCard
              key={manager.id}
              manager={manager}
              portfolio={portfolios[manager.id]}
              positions={positions[manager.id] || []}
              sharpeRatio={MOCK_PERFORMANCE[manager.id]?.sharpe}
              totalReturn={MOCK_PERFORMANCE[manager.id]?.return}
              onClick={() => {
                window.location.href = `/managers/${manager.id}`;
              }}
            />
          ))}
        </div>
      </div>

      {/* Explanation Card */}
      <Card>
        <CardHeader>
          <CardTitle>The Key Question</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <p className="text-muted-foreground">
            Do LLMs add value over systematic quant strategies?
          </p>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="p-4 rounded-lg bg-blue-500/10 border border-blue-500/20">
              <p className="font-medium text-blue-500 mb-2">If LLMs Win</p>
              <p className="text-sm text-muted-foreground">
                AI reasoning creates alpha. LLMs can interpret signals, understand 
                context, and make nuanced decisions that pure algorithms cannot.
              </p>
            </div>
            <div className="p-4 rounded-lg bg-green-500/10 border border-green-500/20">
              <p className="font-medium text-green-500 mb-2">If Quant Bot Wins</p>
              <p className="text-sm text-muted-foreground">
                LLM reasoning doesn&apos;t add value over systematic rules. Simple 
                signal combination with fixed weights outperforms complex AI.
              </p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
