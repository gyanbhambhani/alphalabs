'use client';

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import type { Manager, Portfolio, Position, Trade } from '@/types';
import { cn } from '@/lib/utils';

interface ManagerCardProps {
  manager: Manager;
  portfolio?: Portfolio;
  positions?: Position[];
  recentTrades?: Trade[];
  sharpeRatio?: number;
  totalReturn?: number;
  onClick?: () => void;
}

function formatCurrency(value: number): string {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: 2,
  }).format(value);
}

function formatPercent(value: number): string {
  const formatted = (value * 100).toFixed(2);
  return value >= 0 ? `+${formatted}%` : `${formatted}%`;
}

function getProviderIcon(provider: string | null): string {
  switch (provider) {
    case 'openai':
      return 'GPT';
    case 'anthropic':
      return 'Claude';
    case 'google':
      return 'Gemini';
    default:
      return 'Bot';
  }
}

export function ManagerCard({
  manager,
  portfolio,
  positions = [],
  recentTrades = [],
  sharpeRatio,
  totalReturn,
  onClick,
}: ManagerCardProps) {
  const isQuant = manager.type === 'quant';
  
  return (
    <Card
      className={cn(
        'cursor-pointer transition-all hover:shadow-lg',
        isQuant && 'border-green-500/30 bg-green-500/5'
      )}
      onClick={onClick}
    >
      <CardHeader className="pb-2">
        <CardTitle className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className="text-2xl">
              {isQuant ? '🤖' : '🧠'}
            </span>
            <span>{manager.name}</span>
          </div>
          <Badge
            variant="outline"
            className={cn(
              isQuant
                ? 'bg-green-500/10 text-green-500'
                : 'bg-blue-500/10 text-blue-500'
            )}
          >
            {isQuant ? 'Baseline' : getProviderIcon(manager.provider)}
          </Badge>
        </CardTitle>
      </CardHeader>
      
      <CardContent className="space-y-4">
        {/* Performance Metrics */}
        <div className="grid grid-cols-2 gap-4">
          <div>
            <p className="text-xs text-muted-foreground">Sharpe Ratio</p>
            <p className="text-xl font-mono font-bold">
              {sharpeRatio !== undefined ? sharpeRatio.toFixed(2) : '—'}
            </p>
          </div>
          <div>
            <p className="text-xs text-muted-foreground">Total Return</p>
            <p
              className={cn(
                'text-xl font-mono font-bold',
                totalReturn !== undefined && totalReturn >= 0
                  ? 'text-green-500'
                  : 'text-red-500'
              )}
            >
              {totalReturn !== undefined ? formatPercent(totalReturn) : '—'}
            </p>
          </div>
        </div>
        
        <Separator />
        
        {/* Portfolio Value */}
        <div>
          <p className="text-xs text-muted-foreground">Portfolio Value</p>
          <p className="text-lg font-mono">
            {portfolio ? formatCurrency(portfolio.totalValue) : '—'}
          </p>
          <p className="text-xs text-muted-foreground">
            Cash: {portfolio ? formatCurrency(portfolio.cashBalance) : '—'}
          </p>
        </div>
        
        {/* Current Positions */}
        {positions.length > 0 && (
          <>
            <Separator />
            <div>
              <p className="text-xs text-muted-foreground mb-2">
                Positions ({positions.length})
              </p>
              <div className="space-y-1">
                {positions.slice(0, 3).map((pos) => (
                  <div
                    key={pos.id}
                    className="flex justify-between text-sm font-mono"
                  >
                    <span>{pos.symbol}</span>
                    <span
                      className={cn(
                        pos.unrealizedPnl >= 0 ? 'text-green-500' : 'text-red-500'
                      )}
                    >
                      {formatCurrency(pos.unrealizedPnl)}
                    </span>
                  </div>
                ))}
                {positions.length > 3 && (
                  <p className="text-xs text-muted-foreground">
                    +{positions.length - 3} more...
                  </p>
                )}
              </div>
            </div>
          </>
        )}
        
        {/* Recent Trades */}
        {recentTrades.length > 0 && (
          <>
            <Separator />
            <div>
              <p className="text-xs text-muted-foreground mb-2">Recent Trades</p>
              <div className="space-y-1">
                {recentTrades.slice(0, 2).map((trade) => (
                  <div key={trade.id} className="text-xs">
                    <span
                      className={cn(
                        'font-mono',
                        trade.side === 'buy' ? 'text-green-500' : 'text-red-500'
                      )}
                    >
                      {trade.side.toUpperCase()}
                    </span>{' '}
                    <span className="font-mono">{trade.symbol}</span>
                    {trade.reasoning && (
                      <p className="text-muted-foreground truncate">
                        {trade.reasoning.slice(0, 50)}...
                      </p>
                    )}
                  </div>
                ))}
              </div>
            </div>
          </>
        )}
        
        {/* Manager Type Description */}
        <div className="text-xs text-muted-foreground pt-2 border-t">
          {isQuant ? (
            <p>Pure systematic trading with fixed rules. No LLM reasoning.</p>
          ) : (
            <p>
              Full autonomy to interpret signals and make independent decisions.
            </p>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
