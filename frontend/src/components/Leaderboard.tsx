'use client';

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import type { LeaderboardEntry, Fund, FundStrategy } from '@/types';
import { cn } from '@/lib/utils';

interface LeaderboardProps {
  entries: LeaderboardEntry[];
  funds?: Fund[];
  onSelectManager?: (managerId: string) => void;
  onSelectFund?: (fundId: string) => void;
}

function getRankBadge(rank: number) {
  if (rank === 1) return 'bg-yellow-500 text-black';
  if (rank === 2) return 'bg-gray-400 text-black';
  if (rank === 3) return 'bg-amber-600 text-white';
  return 'bg-muted text-muted-foreground';
}

function getManagerTypeColor(type: string) {
  return type === 'llm' 
    ? 'bg-blue-500/10 text-blue-500' 
    : 'bg-green-500/10 text-green-500';
}

const strategyColors: Record<FundStrategy, string> = {
  trend_macro: 'bg-blue-500/10 text-blue-500',
  mean_reversion: 'bg-green-500/10 text-green-500',
  event_driven: 'bg-orange-500/10 text-orange-500',
  quality_ls: 'bg-purple-500/10 text-purple-500',
};

const strategyLabels: Record<FundStrategy, string> = {
  trend_macro: 'Trend+Macro',
  mean_reversion: 'Mean Rev',
  event_driven: 'Events',
  quality_ls: 'L/S',
};

function formatPercent(value: number): string {
  const formatted = (value * 100).toFixed(2);
  return value >= 0 ? `+${formatted}%` : `${formatted}%`;
}

function formatNumber(value: number, decimals = 2): string {
  return value.toFixed(decimals);
}

function formatCurrency(value: number): string {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: 0,
    maximumFractionDigits: 0,
  }).format(value);
}

export function Leaderboard({ 
  entries, 
  funds = [], 
  onSelectManager, 
  onSelectFund 
}: LeaderboardProps) {
  const hasFunds = funds.length > 0;
  const hasManagers = entries.length > 0;

  // Sort funds by total value
  const sortedFunds = [...funds].sort((a, b) => b.totalValue - a.totalValue);

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          <span>Leaderboard</span>
          {hasFunds ? (
            <Badge variant="outline" className="text-xs">
              Ranked by Portfolio Value
            </Badge>
          ) : (
            <Badge variant="outline" className="text-xs">
              Ranked by Sharpe Ratio
            </Badge>
          )}
        </CardTitle>
      </CardHeader>
      <CardContent>
        {hasFunds ? (
          <Tabs defaultValue="funds" className="w-full">
            <TabsList className="grid w-full grid-cols-2">
              <TabsTrigger value="funds">Funds</TabsTrigger>
              <TabsTrigger value="managers" disabled={!hasManagers}>
                Managers (Legacy)
              </TabsTrigger>
            </TabsList>
            
            {/* Funds Tab */}
            <TabsContent value="funds">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead className="w-12">#</TableHead>
                    <TableHead>Fund</TableHead>
                    <TableHead className="text-right">Value</TableHead>
                    <TableHead className="text-right">P&L</TableHead>
                    <TableHead className="text-right">Gross</TableHead>
                    <TableHead className="text-right">Positions</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {sortedFunds.map((fund, index) => {
                    const pnl = fund.totalValue - 100000;
                    const pnlPct = pnl / 100000;
                    
                    return (
                      <TableRow
                        key={fund.fundId}
                        className={cn(
                          'cursor-pointer hover:bg-muted/50',
                          !fund.isActive && 'opacity-50'
                        )}
                        onClick={() => onSelectFund?.(fund.fundId)}
                      >
                        <TableCell>
                          <Badge className={cn(
                            'w-8 justify-center', 
                            getRankBadge(index + 1)
                          )}>
                            {index + 1}
                          </Badge>
                        </TableCell>
                        <TableCell>
                          <div className="flex items-center gap-2">
                            <span className="font-medium">{fund.name}</span>
                            <Badge 
                              variant="outline" 
                              className={cn(
                                'text-xs', 
                                strategyColors[fund.strategy]
                              )}
                            >
                              {strategyLabels[fund.strategy]}
                            </Badge>
                          </div>
                        </TableCell>
                        <TableCell className="text-right font-mono">
                          {formatCurrency(fund.totalValue)}
                        </TableCell>
                        <TableCell
                          className={cn(
                            'text-right font-mono',
                            pnl >= 0 ? 'text-green-500' : 'text-red-500'
                          )}
                        >
                          {formatPercent(pnlPct)}
                        </TableCell>
                        <TableCell className="text-right font-mono text-muted-foreground">
                          {formatPercent(fund.grossExposure)}
                        </TableCell>
                        <TableCell className="text-right font-mono">
                          {fund.nPositions}
                        </TableCell>
                      </TableRow>
                    );
                  })}
                  {sortedFunds.length === 0 && (
                    <TableRow>
                      <TableCell 
                        colSpan={6} 
                        className="text-center text-muted-foreground"
                      >
                        No funds yet. Create funds to start collaborative trading...
                      </TableCell>
                    </TableRow>
                  )}
                </TableBody>
              </Table>
              
              {sortedFunds.length > 0 && (
                <div className="mt-4 text-xs text-muted-foreground border-t pt-4">
                  <p>
                    <strong>Collaborative AI:</strong> Multiple AI models debate 
                    and collaborate within each fund. PM Finalizer makes the call.
                  </p>
                </div>
              )}
            </TabsContent>

            {/* Legacy Managers Tab */}
            <TabsContent value="managers">
              <ManagerTable entries={entries} onSelectManager={onSelectManager} />
            </TabsContent>
          </Tabs>
        ) : (
          <ManagerTable entries={entries} onSelectManager={onSelectManager} />
        )}
      </CardContent>
    </Card>
  );
}

// Extracted manager table component
function ManagerTable({ 
  entries, 
  onSelectManager 
}: { 
  entries: LeaderboardEntry[]; 
  onSelectManager?: (id: string) => void; 
}) {
  return (
    <>
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead className="w-12">#</TableHead>
            <TableHead>Manager</TableHead>
            <TableHead className="text-right">Sharpe</TableHead>
            <TableHead className="text-right">Return</TableHead>
            <TableHead className="text-right">Vol</TableHead>
            <TableHead className="text-right">Max DD</TableHead>
            <TableHead className="text-right">Trades</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {entries.length === 0 ? (
            <TableRow>
              <TableCell colSpan={7} className="text-center text-muted-foreground">
                No data yet. Waiting for trading to begin...
              </TableCell>
            </TableRow>
          ) : (
            entries.map((entry) => (
              <TableRow
                key={entry.manager.id}
                className={cn(
                  'cursor-pointer hover:bg-muted/50',
                  entry.manager.type === 'quant' && 'bg-green-500/5'
                )}
                onClick={() => onSelectManager?.(entry.manager.id)}
              >
                <TableCell>
                  <Badge className={cn('w-8 justify-center', getRankBadge(entry.rank))}>
                    {entry.rank}
                  </Badge>
                </TableCell>
                <TableCell>
                  <div className="flex items-center gap-2">
                    <span className="font-medium">{entry.manager.name}</span>
                    <Badge 
                      variant="outline" 
                      className={cn('text-xs', getManagerTypeColor(entry.manager.type))}
                    >
                      {entry.manager.type === 'llm' ? entry.manager.provider : 'Quant'}
                    </Badge>
                  </div>
                </TableCell>
                <TableCell className="text-right font-mono">
                  {formatNumber(entry.sharpeRatio)}
                </TableCell>
                <TableCell
                  className={cn(
                    'text-right font-mono',
                    entry.totalReturn >= 0 ? 'text-green-500' : 'text-red-500'
                  )}
                >
                  {formatPercent(entry.totalReturn)}
                </TableCell>
                <TableCell className="text-right font-mono text-muted-foreground">
                  {formatPercent(entry.volatility)}
                </TableCell>
                <TableCell className="text-right font-mono text-red-500">
                  {formatPercent(entry.maxDrawdown)}
                </TableCell>
                <TableCell className="text-right font-mono">
                  {entry.totalTrades}
                </TableCell>
              </TableRow>
            ))
          )}
        </TableBody>
      </Table>
      
      {entries.length > 0 && (
        <div className="mt-4 text-xs text-muted-foreground border-t pt-4">
          <p>
            <strong>Key Question:</strong> If Quant Bot beats LLMs → LLM reasoning 
            doesn&apos;t add value. If LLMs beat Quant Bot → AI reasoning creates alpha.
          </p>
        </div>
      )}
    </>
  );
}
