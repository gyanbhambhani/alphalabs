'use client';

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import type { Trade } from '@/types';
import { cn } from '@/lib/utils';
import { formatDistanceToNow } from 'date-fns';

interface TradeHistoryProps {
  trades: Trade[];
  managerNames?: Record<string, string>;
  showManager?: boolean;
}

function formatCurrency(value: number): string {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: 2,
  }).format(value);
}

export function TradeHistory({ 
  trades, 
  managerNames = {}, 
  showManager = true 
}: TradeHistoryProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Recent Trades</CardTitle>
      </CardHeader>
      <CardContent>
        {trades.length === 0 ? (
          <p className="text-muted-foreground text-sm">
            No trades yet. Waiting for trading to begin...
          </p>
        ) : (
          <div className="space-y-3">
            {trades.map((trade) => (
              <div
                key={trade.id}
                className="p-3 rounded-lg bg-muted/50 space-y-2"
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <Badge
                      className={cn(
                        trade.side === 'buy'
                          ? 'bg-green-500/20 text-green-500'
                          : 'bg-red-500/20 text-red-500'
                      )}
                    >
                      {trade.side.toUpperCase()}
                    </Badge>
                    <span className="font-mono font-bold">{trade.symbol}</span>
                    <span className="text-sm text-muted-foreground">
                      {trade.quantity} @ {formatCurrency(trade.price)}
                    </span>
                  </div>
                  {showManager && managerNames[trade.managerId] && (
                    <Badge variant="outline">
                      {managerNames[trade.managerId]}
                    </Badge>
                  )}
                </div>
                
                {trade.reasoning && (
                  <p className="text-xs text-muted-foreground">
                    {trade.reasoning}
                  </p>
                )}
                
                <div className="flex items-center justify-between text-xs text-muted-foreground">
                  <span>
                    {formatDistanceToNow(new Date(trade.executedAt), { 
                      addSuffix: true 
                    })}
                  </span>
                  {trade.signalsUsed && (
                    <span>
                      Signals: {Object.keys(trade.signalsUsed).join(', ')}
                    </span>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
