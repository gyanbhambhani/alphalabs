'use client';

import { useEffect, useState } from 'react';
import Link from 'next/link';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Skeleton } from '@/components/ui/skeleton';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import { FundCard } from '@/components/FundCard';
import { api } from '@/lib/api';
import type { Fund, FundStrategy, DecisionRecord } from '@/types';
import { cn } from '@/lib/utils';

const strategyColors: Record<FundStrategy, string> = {
  trend_macro: 'bg-blue-500',
  mean_reversion: 'bg-green-500',
  event_driven: 'bg-orange-500',
  quality_ls: 'bg-purple-500',
};

const strategyLabels: Record<FundStrategy, string> = {
  trend_macro: 'Trend + Macro',
  mean_reversion: 'Mean Reversion',
  event_driven: 'Event-Driven',
  quality_ls: 'Quality L/S',
};

function formatCurrency(value: number): string {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: 0,
    maximumFractionDigits: 0,
  }).format(value);
}

function formatPercent(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

function getRankBadge(rank: number) {
  if (rank === 1) return 'bg-yellow-500 text-black';
  if (rank === 2) return 'bg-gray-400 text-black';
  if (rank === 3) return 'bg-amber-600 text-white';
  return 'bg-muted text-muted-foreground';
}

export default function FundsPage() {
  const [funds, setFunds] = useState<Fund[]>([]);
  const [allDecisions, setAllDecisions] = useState<DecisionRecord[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [viewMode, setViewMode] = useState<'cards' | 'table'>('cards');

  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    setIsLoading(true);
    try {
      const fundsData = await api.getFunds();
      setFunds(fundsData);

      // Load decisions for all funds
      const decisionsPromises = fundsData.map(f => 
        api.getFundDecisions(f.fundId, 10).catch(() => [])
      );
      const decisionsArrays = await Promise.all(decisionsPromises);
      const allDec = decisionsArrays.flat().sort((a, b) => 
        new Date(b.asofTimestamp).getTime() - new Date(a.asofTimestamp).getTime()
      );
      setAllDecisions(allDec.slice(0, 20));
    } catch (error) {
      console.error('Failed to load funds:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const sortedFunds = [...funds].sort((a, b) => b.totalValue - a.totalValue);

  // Calculate totals
  const totalAUM = funds.reduce((sum, f) => sum + f.totalValue, 0);
  const totalPnL = funds.reduce((sum, f) => sum + (f.totalValue - 100000), 0);
  const avgGross = funds.length > 0 
    ? funds.reduce((sum, f) => sum + f.grossExposure, 0) / funds.length 
    : 0;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-2xl font-bold mb-2">Collaborative Funds</h1>
          <p className="text-muted-foreground">
            Thesis-driven funds where multiple AI models collaborate via structured debate
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Button 
            variant={viewMode === 'cards' ? 'default' : 'outline'} 
            size="sm"
            onClick={() => setViewMode('cards')}
          >
            Cards
          </Button>
          <Button 
            variant={viewMode === 'table' ? 'default' : 'outline'} 
            size="sm"
            onClick={() => setViewMode('table')}
          >
            Table
          </Button>
        </div>
      </div>

      {/* Summary Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm text-muted-foreground">Total AUM</CardTitle>
          </CardHeader>
          <CardContent>
            {isLoading ? (
              <Skeleton className="h-8 w-24" />
            ) : (
              <p className="text-2xl font-bold">{formatCurrency(totalAUM)}</p>
            )}
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm text-muted-foreground">Total P&L</CardTitle>
          </CardHeader>
          <CardContent>
            {isLoading ? (
              <Skeleton className="h-8 w-24" />
            ) : (
              <p className={cn(
                "text-2xl font-bold",
                totalPnL >= 0 ? "text-green-500" : "text-red-500"
              )}>
                {totalPnL >= 0 ? '+' : ''}{formatCurrency(totalPnL)}
              </p>
            )}
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm text-muted-foreground">Avg Gross Exposure</CardTitle>
          </CardHeader>
          <CardContent>
            {isLoading ? (
              <Skeleton className="h-8 w-20" />
            ) : (
              <p className="text-2xl font-bold">{formatPercent(avgGross)}</p>
            )}
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm text-muted-foreground">Active Funds</CardTitle>
          </CardHeader>
          <CardContent>
            {isLoading ? (
              <Skeleton className="h-8 w-16" />
            ) : (
              <p className="text-2xl font-bold">
                {funds.filter(f => f.isActive).length} / {funds.length}
              </p>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Main Content */}
      <Tabs defaultValue="funds" className="w-full">
        <TabsList>
          <TabsTrigger value="funds">All Funds</TabsTrigger>
          <TabsTrigger value="activity">Recent Activity</TabsTrigger>
        </TabsList>

        <TabsContent value="funds" className="mt-4">
          {isLoading ? (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {[1, 2, 3, 4].map(i => (
                <Card key={i}>
                  <CardContent className="p-6">
                    <Skeleton className="h-40 w-full" />
                  </CardContent>
                </Card>
              ))}
            </div>
          ) : viewMode === 'cards' ? (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {sortedFunds.map((fund, index) => (
                <div key={fund.fundId} className="relative">
                  <div className="absolute -top-2 -left-2 w-8 h-8 rounded-full bg-primary text-primary-foreground flex items-center justify-center font-bold text-sm z-10">
                    #{index + 1}
                  </div>
                  <Link href={`/funds/${fund.fundId}`}>
                    <FundCard fund={fund} />
                  </Link>
                </div>
              ))}
              {sortedFunds.length === 0 && (
                <div className="col-span-full">
                  <Card>
                    <CardContent className="py-12 text-center text-muted-foreground">
                      No funds created yet. Initialize the system to create funds.
                    </CardContent>
                  </Card>
                </div>
              )}
            </div>
          ) : (
            <Card>
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead className="w-12">#</TableHead>
                    <TableHead>Fund</TableHead>
                    <TableHead className="text-right">Value</TableHead>
                    <TableHead className="text-right">P&L</TableHead>
                    <TableHead className="text-right">Gross</TableHead>
                    <TableHead className="text-right">Positions</TableHead>
                    <TableHead className="text-right">Status</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {sortedFunds.map((fund, index) => {
                    const pnl = fund.totalValue - 100000;
                    return (
                      <TableRow 
                        key={fund.fundId}
                        className="cursor-pointer hover:bg-muted/50"
                        onClick={() => window.location.href = `/funds/${fund.fundId}`}
                      >
                        <TableCell>
                          <Badge className={cn('w-8 justify-center', getRankBadge(index + 1))}>
                            {index + 1}
                          </Badge>
                        </TableCell>
                        <TableCell>
                          <div className="flex items-center gap-2">
                            <span className="font-medium">{fund.name}</span>
                            <Badge className={cn('text-xs text-white', strategyColors[fund.strategy])}>
                              {strategyLabels[fund.strategy]}
                            </Badge>
                          </div>
                        </TableCell>
                        <TableCell className="text-right font-mono">
                          {formatCurrency(fund.totalValue)}
                        </TableCell>
                        <TableCell className={cn(
                          "text-right font-mono",
                          pnl >= 0 ? "text-green-500" : "text-red-500"
                        )}>
                          {pnl >= 0 ? '+' : ''}{formatPercent(pnl / 100000)}
                        </TableCell>
                        <TableCell className="text-right font-mono text-muted-foreground">
                          {formatPercent(fund.grossExposure)}
                        </TableCell>
                        <TableCell className="text-right font-mono">
                          {fund.nPositions}
                        </TableCell>
                        <TableCell className="text-right">
                          <Badge variant={fund.isActive ? 'default' : 'secondary'}>
                            {fund.isActive ? 'Active' : 'Inactive'}
                          </Badge>
                        </TableCell>
                      </TableRow>
                    );
                  })}
                </TableBody>
              </Table>
            </Card>
          )}
        </TabsContent>

        <TabsContent value="activity" className="mt-4">
          <Card>
            <CardHeader>
              <CardTitle>Recent Decisions Across All Funds</CardTitle>
            </CardHeader>
            <CardContent>
              {isLoading ? (
                <div className="space-y-2">
                  {[1, 2, 3, 4, 5].map(i => (
                    <Skeleton key={i} className="h-12 w-full" />
                  ))}
                </div>
              ) : allDecisions.length > 0 ? (
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Time</TableHead>
                      <TableHead>Fund</TableHead>
                      <TableHead>Type</TableHead>
                      <TableHead>Status</TableHead>
                      <TableHead>Details</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {allDecisions.map(decision => {
                      const fund = funds.find(f => f.fundId === decision.fundId);
                      return (
                        <TableRow 
                          key={decision.decisionId}
                          className="cursor-pointer hover:bg-muted/50"
                          onClick={() => window.location.href = `/funds/${decision.fundId}?decision=${decision.decisionId}`}
                        >
                          <TableCell className="font-mono text-sm">
                            {new Date(decision.asofTimestamp).toLocaleString()}
                          </TableCell>
                          <TableCell>
                            <Badge variant="outline">
                              {fund?.name || decision.fundId}
                            </Badge>
                          </TableCell>
                          <TableCell>
                            <Badge variant={decision.decisionType === 'trade' ? 'default' : 'secondary'}>
                              {decision.decisionType}
                            </Badge>
                          </TableCell>
                          <TableCell>
                            <Badge variant="outline">
                              {decision.status}
                            </Badge>
                          </TableCell>
                          <TableCell className="text-sm text-muted-foreground">
                            {decision.noTradeReason || 
                              (decision.predictedDirections 
                                ? `${Object.keys(decision.predictedDirections).length} positions`
                                : '-'
                              )}
                          </TableCell>
                        </TableRow>
                      );
                    })}
                  </TableBody>
                </Table>
              ) : (
                <p className="text-center text-muted-foreground py-8">
                  No decisions yet. Run a trading cycle to generate activity.
                </p>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
