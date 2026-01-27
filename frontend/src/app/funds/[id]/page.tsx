'use client';

import { useEffect, useState } from 'react';
import { useParams, useSearchParams } from 'next/navigation';
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
import { DebateViewer, DecisionList } from '@/components/DebateViewer';
import { api } from '@/lib/api';
import type { 
  FundDetail, 
  FundPosition, 
  DecisionRecord, 
  DecisionDetail,
  DebateTranscript,
  FundStrategy 
} from '@/types';
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
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  }).format(value);
}

function formatPercent(value: number): string {
  return `${(value * 100).toFixed(2)}%`;
}

export default function FundDetailPage() {
  const params = useParams();
  const searchParams = useSearchParams();
  const fundId = params.id as string;
  const selectedDecisionId = searchParams.get('decision');

  const [fund, setFund] = useState<FundDetail | null>(null);
  const [positions, setPositions] = useState<FundPosition[]>([]);
  const [decisions, setDecisions] = useState<DecisionRecord[]>([]);
  const [selectedDecision, setSelectedDecision] = useState<DecisionDetail | null>(null);
  const [selectedTranscript, setSelectedTranscript] = useState<DebateTranscript | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    loadData();
  }, [fundId]);

  useEffect(() => {
    if (selectedDecisionId) {
      loadDecisionDetail(selectedDecisionId);
    }
  }, [selectedDecisionId]);

  const loadData = async () => {
    setIsLoading(true);
    try {
      const [fundData, posData, decData] = await Promise.all([
        api.getFund(fundId),
        api.getFundPositions(fundId),
        api.getFundDecisions(fundId, 50),
      ]);
      setFund(fundData);
      setPositions(posData);
      setDecisions(decData);
    } catch (error) {
      console.error('Failed to load fund data:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const loadDecisionDetail = async (decisionId: string) => {
    try {
      const [detail, transcript] = await Promise.all([
        api.getDecision(decisionId),
        api.getDecisionDebate(decisionId).catch(() => null),
      ]);
      setSelectedDecision(detail);
      setSelectedTranscript(transcript);
    } catch (error) {
      console.error('Failed to load decision:', error);
    }
  };

  const handleDecisionClick = (decisionId: string) => {
    window.history.pushState({}, '', `/funds/${fundId}?decision=${decisionId}`);
    loadDecisionDetail(decisionId);
  };

  const handleCloseDecision = () => {
    window.history.pushState({}, '', `/funds/${fundId}`);
    setSelectedDecision(null);
    setSelectedTranscript(null);
  };

  if (isLoading) {
    return (
      <div className="space-y-6">
        <Skeleton className="h-12 w-64" />
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          {[1, 2, 3, 4].map(i => (
            <Card key={i}><CardContent className="p-6"><Skeleton className="h-20" /></CardContent></Card>
          ))}
        </div>
        <Skeleton className="h-96 w-full" />
      </div>
    );
  }

  if (!fund) {
    return (
      <div className="text-center py-12">
        <p className="text-muted-foreground">Fund not found</p>
        <Link href="/funds">
          <Button variant="outline" className="mt-4">Back to Funds</Button>
        </Link>
      </div>
    );
  }

  const pnl = fund.totalValue - 100000;
  const pnlPct = pnl / 100000;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          <div className="flex items-center gap-3 mb-2">
            <Link href="/funds" className="text-muted-foreground hover:text-foreground">
              ← Funds
            </Link>
          </div>
          <h1 className="text-2xl font-bold flex items-center gap-3">
            {fund.name}
            <Badge className={cn('text-white', strategyColors[fund.strategy])}>
              {strategyLabels[fund.strategy]}
            </Badge>
            {!fund.isActive && (
              <Badge variant="secondary">Inactive</Badge>
            )}
          </h1>
          <p className="text-muted-foreground mt-1">
            {fund.description || fund.thesis?.description || 'No description'}
          </p>
        </div>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
        <Card>
          <CardHeader className="pb-1">
            <CardTitle className="text-xs text-muted-foreground">Total Value</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-xl font-bold">{formatCurrency(fund.totalValue)}</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-1">
            <CardTitle className="text-xs text-muted-foreground">P&L</CardTitle>
          </CardHeader>
          <CardContent>
            <p className={cn("text-xl font-bold", pnl >= 0 ? "text-green-500" : "text-red-500")}>
              {pnl >= 0 ? '+' : ''}{formatCurrency(pnl)}
              <span className="text-sm ml-1">({formatPercent(pnlPct)})</span>
            </p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-1">
            <CardTitle className="text-xs text-muted-foreground">Cash</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-xl font-bold">{formatCurrency(fund.cashBalance)}</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-1">
            <CardTitle className="text-xs text-muted-foreground">Gross Exposure</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-xl font-bold">{formatPercent(fund.grossExposure)}</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-1">
            <CardTitle className="text-xs text-muted-foreground">Positions</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-xl font-bold">{fund.nPositions}</p>
          </CardContent>
        </Card>
      </div>

      {/* Selected Decision Detail */}
      {selectedDecision && (
        <DebateViewer
          decision={selectedDecision}
          transcript={selectedTranscript || undefined}
          onClose={handleCloseDecision}
        />
      )}

      {/* Main Content */}
      <Tabs defaultValue="positions" className="w-full">
        <TabsList>
          <TabsTrigger value="positions">Positions ({positions.length})</TabsTrigger>
          <TabsTrigger value="decisions">Decisions ({decisions.length})</TabsTrigger>
          <TabsTrigger value="config">Configuration</TabsTrigger>
        </TabsList>

        {/* Positions Tab */}
        <TabsContent value="positions" className="mt-4">
          <Card>
            <CardHeader>
              <CardTitle>Current Positions</CardTitle>
            </CardHeader>
            <CardContent>
              {positions.length > 0 ? (
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Symbol</TableHead>
                      <TableHead className="text-right">Quantity</TableHead>
                      <TableHead className="text-right">Avg Entry</TableHead>
                      <TableHead className="text-right">Current</TableHead>
                      <TableHead className="text-right">Market Value</TableHead>
                      <TableHead className="text-right">P&L</TableHead>
                      <TableHead className="text-right">Weight</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {positions.map(pos => (
                      <TableRow key={pos.symbol}>
                        <TableCell className="font-medium">{pos.symbol}</TableCell>
                        <TableCell className="text-right font-mono">
                          {pos.quantity.toFixed(2)}
                        </TableCell>
                        <TableCell className="text-right font-mono">
                          ${pos.avgEntryPrice.toFixed(2)}
                        </TableCell>
                        <TableCell className="text-right font-mono">
                          ${pos.currentPrice.toFixed(2)}
                        </TableCell>
                        <TableCell className="text-right font-mono">
                          {formatCurrency(pos.marketValue)}
                        </TableCell>
                        <TableCell className={cn(
                          "text-right font-mono",
                          pos.unrealizedPnl >= 0 ? "text-green-500" : "text-red-500"
                        )}>
                          {pos.unrealizedPnl >= 0 ? '+' : ''}{formatCurrency(pos.unrealizedPnl)}
                        </TableCell>
                        <TableCell className="text-right font-mono">
                          {pos.weightPct.toFixed(1)}%
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              ) : (
                <p className="text-center text-muted-foreground py-8">
                  No positions. The fund is fully in cash.
                </p>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* Decisions Tab */}
        <TabsContent value="decisions" className="mt-4">
          <Card>
            <CardHeader>
              <CardTitle>Decision History</CardTitle>
            </CardHeader>
            <CardContent>
              {decisions.length > 0 ? (
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Time</TableHead>
                      <TableHead>Type</TableHead>
                      <TableHead>Status</TableHead>
                      <TableHead>Details</TableHead>
                      <TableHead>Inputs Hash</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {decisions.map(decision => (
                      <TableRow 
                        key={decision.decisionId}
                        className={cn(
                          "cursor-pointer hover:bg-muted/50",
                          selectedDecisionId === decision.decisionId && "bg-muted"
                        )}
                        onClick={() => handleDecisionClick(decision.decisionId)}
                      >
                        <TableCell className="font-mono text-sm">
                          {new Date(decision.asofTimestamp).toLocaleString()}
                        </TableCell>
                        <TableCell>
                          <Badge variant={decision.decisionType === 'trade' ? 'default' : 'secondary'}>
                            {decision.decisionType}
                          </Badge>
                        </TableCell>
                        <TableCell>
                          <Badge variant="outline">{decision.status}</Badge>
                        </TableCell>
                        <TableCell className="text-sm">
                          {decision.noTradeReason || 
                            (decision.predictedDirections 
                              ? `${Object.keys(decision.predictedDirections).length} positions`
                              : '-'
                            )}
                        </TableCell>
                        <TableCell className="font-mono text-xs text-muted-foreground">
                          {decision.inputsHash || '-'}
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              ) : (
                <p className="text-center text-muted-foreground py-8">
                  No decisions yet. Run a trading cycle to generate decisions.
                </p>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* Config Tab */}
        <TabsContent value="config" className="mt-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* Thesis */}
            <Card>
              <CardHeader>
                <CardTitle>Fund Thesis</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                {fund.thesis ? (
                  <>
                    <div>
                      <p className="text-sm text-muted-foreground">Strategy</p>
                      <p className="font-medium">{fund.thesis.strategy}</p>
                    </div>
                    <div>
                      <p className="text-sm text-muted-foreground">Horizon</p>
                      <p className="font-medium">
                        {fund.thesis.horizonDays[0]} - {fund.thesis.horizonDays[1]} days
                      </p>
                    </div>
                    <div>
                      <p className="text-sm text-muted-foreground">Edge</p>
                      <p className="font-medium">{fund.thesis.edge}</p>
                    </div>
                    <div>
                      <p className="text-sm text-muted-foreground">Universe</p>
                      <p className="font-medium font-mono text-sm">
                        {fund.thesis.universeSpec?.type}: {JSON.stringify(fund.thesis.universeSpec?.params)}
                      </p>
                    </div>
                  </>
                ) : (
                  <p className="text-muted-foreground">No thesis configuration available</p>
                )}
              </CardContent>
            </Card>

            {/* Policy */}
            <Card>
              <CardHeader>
                <CardTitle>Fund Policy</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                {fund.policy ? (
                  <>
                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <p className="text-sm text-muted-foreground">Sizing Method</p>
                        <p className="font-medium">{fund.policy.sizingMethod}</p>
                      </div>
                      <div>
                        <p className="text-sm text-muted-foreground">Vol Target</p>
                        <p className="font-medium">
                          {fund.policy.volTarget ? formatPercent(fund.policy.volTarget) : '-'}
                        </p>
                      </div>
                      <div>
                        <p className="text-sm text-muted-foreground">Max Position</p>
                        <p className="font-medium">{formatPercent(fund.policy.maxPositionPct)}</p>
                      </div>
                      <div>
                        <p className="text-sm text-muted-foreground">Max Positions</p>
                        <p className="font-medium">{fund.policy.maxPositions}</p>
                      </div>
                      <div>
                        <p className="text-sm text-muted-foreground">Max Gross</p>
                        <p className="font-medium">{formatPercent(fund.policy.maxGrossExposure)}</p>
                      </div>
                      <div>
                        <p className="text-sm text-muted-foreground">Min Cash</p>
                        <p className="font-medium">{formatPercent(fund.policy.minCashBuffer)}</p>
                      </div>
                    </div>
                    <div className="pt-2 border-t">
                      <p className="text-sm text-muted-foreground">Exit Rules</p>
                      <p className="font-medium">
                        SL: {formatPercent(fund.policy.defaultStopLossPct)} | 
                        TP: {formatPercent(fund.policy.defaultTakeProfitPct)}
                        {fund.policy.trailingStop && ' | Trailing'}
                      </p>
                    </div>
                  </>
                ) : (
                  <p className="text-muted-foreground">No policy configuration available</p>
                )}
              </CardContent>
            </Card>

            {/* Risk Limits */}
            <Card className="md:col-span-2">
              <CardHeader>
                <CardTitle>Risk Limits</CardTitle>
              </CardHeader>
              <CardContent>
                {fund.riskLimits ? (
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    <div>
                      <p className="text-sm text-muted-foreground">Max Daily Loss</p>
                      <p className="font-medium text-red-500">
                        -{formatPercent(fund.riskLimits.maxDailyLossPct)}
                      </p>
                    </div>
                    <div>
                      <p className="text-sm text-muted-foreground">Max Weekly DD</p>
                      <p className="font-medium text-red-500">
                        -{formatPercent(fund.riskLimits.maxWeeklyDrawdownPct)}
                      </p>
                    </div>
                    <div>
                      <p className="text-sm text-muted-foreground">Breach Action</p>
                      <p className="font-medium">{fund.riskLimits.breachAction}</p>
                    </div>
                    <div>
                      <p className="text-sm text-muted-foreground">Cooldown</p>
                      <p className="font-medium">{fund.riskLimits.breachCooldownDays} days</p>
                    </div>
                  </div>
                ) : (
                  <p className="text-muted-foreground">No risk limits configuration available</p>
                )}
              </CardContent>
            </Card>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
}
