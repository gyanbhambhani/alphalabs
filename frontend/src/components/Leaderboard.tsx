'use client';

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import type { LeaderboardEntry } from '@/types';
import { cn } from '@/lib/utils';

interface LeaderboardProps {
  entries: LeaderboardEntry[];
  onSelectManager?: (managerId: string) => void;
}

function getRankBadge(rank: number) {
  if (rank === 1) return 'bg-yellow-500 text-black';
  if (rank === 2) return 'bg-gray-400 text-black';
  if (rank === 3) return 'bg-amber-600 text-white';
  return 'bg-muted text-muted-foreground';
}

function getManagerTypeColor(type: string) {
  return type === 'llm' ? 'bg-blue-500/10 text-blue-500' : 'bg-green-500/10 text-green-500';
}

function formatPercent(value: number): string {
  const formatted = (value * 100).toFixed(2);
  return value >= 0 ? `+${formatted}%` : `${formatted}%`;
}

function formatNumber(value: number, decimals = 2): string {
  return value.toFixed(decimals);
}

export function Leaderboard({ entries, onSelectManager }: LeaderboardProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          <span>Leaderboard</span>
          <Badge variant="outline" className="text-xs">
            Ranked by Sharpe Ratio
          </Badge>
        </CardTitle>
      </CardHeader>
      <CardContent>
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
      </CardContent>
    </Card>
  );
}
