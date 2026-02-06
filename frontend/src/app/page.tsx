'use client';

import { useEffect, useState } from 'react';
import Link from 'next/link';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Skeleton } from '@/components/ui/skeleton';
import { FundCard } from '@/components/FundCard';
import { DecisionQueue } from '@/components/DecisionQueue';
import { EnhancedDecisionViewer } from '@/components/EnhancedDecisionViewer';
import { api } from '@/lib/api';
import type { Fund, DecisionRecord, FundStrategy } from '@/types';

const strategyDescriptions: Record<FundStrategy, string> = {
  trend_macro: 'Regime detection + trend following',
  mean_reversion: 'Exploit overreactions in liquid names',
  event_driven: 'Earnings plays and event catalysts',
  quality_ls: 'Fundamental long-short strategies',
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

export default function Dashboard() {
  const [funds, setFunds] = useState<Fund[]>([]);
  const [recentDecisions, setRecentDecisions] = useState<DecisionRecord[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [isConnected, setIsConnected] = useState(false);
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null);
  const [isTriggeringCycle, setIsTriggeringCycle] = useState(false);
  const [selectedDecision, setSelectedDecision] = useState<string | null>(null);
  const [showDecisionModal, setShowDecisionModal] = useState(false);

  useEffect(() => {
    loadData();
    // Poll every 30 seconds
    const interval = setInterval(loadData, 30000);
    return () => clearInterval(interval);
  }, []);

  const loadData = async () => {
    try {
      // Check health first
      await api.healthCheck();
      setIsConnected(true);

      // Load funds
      const fundsData = await api.getFunds();
      setFunds(fundsData);

      // Load recent decisions from first fund (if any)
      if (fundsData.length > 0) {
        // Get decisions from all funds
        const allDecisions = await Promise.all(
          fundsData.map(fund => api.getFundDecisions(fund.fundId, 3))
        );
        const flatDecisions = allDecisions
          .flat()
          .sort((a, b) => 
            new Date(b.asofTimestamp).getTime() - 
            new Date(a.asofTimestamp).getTime()
          )
          .slice(0, 10);
        setRecentDecisions(flatDecisions);
      }

      setLastUpdate(new Date());
    } catch (error) {
      console.error('Failed to load data:', error);
      setIsConnected(false);
    } finally {
      setIsLoading(false);
    }
  };

  const handleTriggerCycle = async () => {
    setIsTriggeringCycle(true);
    try {
      const result = await api.triggerFundTradingCycle();
      alert(result.message);
      loadData();
    } catch (error) {
      alert('Failed to trigger trading cycle');
    } finally {
      setIsTriggeringCycle(false);
    }
  };

  // Calculate totals
  const totalAUM = funds.reduce((sum, f) => sum + f.totalValue, 0);
  const totalPositions = funds.reduce((sum, f) => sum + f.nPositions, 0);
  const activeFunds = funds.filter(f => f.isActive).length;

  // Sort funds by value for display
  const sortedFunds = [...funds].sort((a, b) => b.totalValue - a.totalValue);

  return (
    <div className="space-y-8">
      {/* Hero Section */}
      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-3xl font-bold mb-2">Trading Dashboard</h1>
          <p className="text-muted-foreground max-w-2xl">
            Multiple AI models collaborate within thesis-driven funds. Each decision 
            goes through a structured debate: propose, critique, synthesize, risk check, 
            and PM finalization.
          </p>
        </div>
        <div className="flex items-center gap-3">
          <Badge
            variant="outline"
            className={
              isConnected
                ? 'bg-green-500/20 text-green-500 border-green-500/50'
                : 'bg-red-500/20 text-red-500 border-red-500/50'
            }
          >
            {isConnected ? '● Connected' : '○ Disconnected'}
          </Badge>
          <Button 
            onClick={handleTriggerCycle}
            disabled={isTriggeringCycle || !isConnected}
            size="sm"
          >
            {isTriggeringCycle ? 'Running...' : 'Run Trading Cycle'}
          </Button>
        </div>
      </div>

      {/* Stats Overview */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm text-muted-foreground">
              Total AUM
            </CardTitle>
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
            <CardTitle className="text-sm text-muted-foreground">
              Active Funds
            </CardTitle>
          </CardHeader>
          <CardContent>
            {isLoading ? (
              <Skeleton className="h-8 w-16" />
            ) : (
              <p className="text-2xl font-bold">{activeFunds} / {funds.length}</p>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm text-muted-foreground">
              Total Positions
            </CardTitle>
          </CardHeader>
          <CardContent>
            {isLoading ? (
              <Skeleton className="h-8 w-16" />
            ) : (
              <p className="text-2xl font-bold">{totalPositions}</p>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm text-muted-foreground">
              Last Update
            </CardTitle>
          </CardHeader>
          <CardContent>
            {isLoading ? (
              <Skeleton className="h-8 w-24" />
            ) : (
              <p className="text-sm font-mono">
                {lastUpdate ? lastUpdate.toLocaleTimeString() : '-'}
              </p>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Main Content */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Funds Grid */}
        <div className="lg:col-span-2 space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-xl font-semibold">Funds</h2>
            <Link href="/funds">
              <Button variant="outline" size="sm">View All</Button>
            </Link>
          </div>
          
          {isLoading ? (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {[1, 2, 3, 4].map(i => (
                <Card key={i}>
                  <CardContent className="p-6">
                    <Skeleton className="h-32 w-full" />
                  </CardContent>
                </Card>
              ))}
            </div>
          ) : sortedFunds.length > 0 ? (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {sortedFunds.slice(0, 4).map((fund, index) => (
                <div key={fund.fundId} className="relative">
                  <div className="absolute -top-2 -left-2 w-7 h-7 rounded-full bg-primary text-primary-foreground flex items-center justify-center font-bold text-xs z-10">
                    #{index + 1}
                  </div>
                  <Link href={`/funds/${fund.fundId}`}>
                    <FundCard fund={fund} />
                  </Link>
                </div>
              ))}
            </div>
          ) : (
            <Card>
              <CardContent className="py-12 text-center">
                <p className="text-muted-foreground mb-4">
                  No funds created yet. Funds will appear here once the system is initialized.
                </p>
                <p className="text-sm text-muted-foreground">
                  Each fund represents a thesis-driven strategy where multiple AI models collaborate.
                </p>
              </CardContent>
            </Card>
          )}
        </div>

        {/* Sidebar */}
        <div className="space-y-6">
          {/* Recent Decisions - Enhanced */}
          <DecisionQueue
            decisions={recentDecisions}
            isLoading={isLoading}
            onDecisionClick={(id) => {
              setSelectedDecision(id);
              setShowDecisionModal(true);
            }}
            showFilters={false}
            enableStreaming={false}
          />

          {/* How It Works */}
          <Card>
            <CardHeader>
              <CardTitle className="text-lg">How It Works</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="flex items-start gap-3">
                <div className="w-6 h-6 rounded-full bg-blue-500/20 text-blue-500 flex items-center justify-center text-xs font-bold">1</div>
                <div>
                  <p className="font-medium text-sm">Propose</p>
                  <p className="text-xs text-muted-foreground">AI models propose trade candidates</p>
                </div>
              </div>
              <div className="flex items-start gap-3">
                <div className="w-6 h-6 rounded-full bg-orange-500/20 text-orange-500 flex items-center justify-center text-xs font-bold">2</div>
                <div>
                  <p className="font-medium text-sm">Critique</p>
                  <p className="text-xs text-muted-foreground">Models critique each other&apos;s proposals</p>
                </div>
              </div>
              <div className="flex items-start gap-3">
                <div className="w-6 h-6 rounded-full bg-purple-500/20 text-purple-500 flex items-center justify-center text-xs font-bold">3</div>
                <div>
                  <p className="font-medium text-sm">Synthesize</p>
                  <p className="text-xs text-muted-foreground">Merge proposals into consensus plans</p>
                </div>
              </div>
              <div className="flex items-start gap-3">
                <div className="w-6 h-6 rounded-full bg-red-500/20 text-red-500 flex items-center justify-center text-xs font-bold">4</div>
                <div>
                  <p className="font-medium text-sm">Risk Check</p>
                  <p className="text-xs text-muted-foreground">Risk Manager validates or vetoes</p>
                </div>
              </div>
              <div className="flex items-start gap-3">
                <div className="w-6 h-6 rounded-full bg-green-500/20 text-green-500 flex items-center justify-center text-xs font-bold">5</div>
                <div>
                  <p className="font-medium text-sm">Finalize</p>
                  <p className="text-xs text-muted-foreground">PM Finalizer makes the call</p>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Fund Strategies */}
          <Card>
            <CardHeader>
              <CardTitle className="text-lg">Fund Strategies</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              {Object.entries(strategyDescriptions).map(([strategy, desc]) => (
                <div key={strategy} className="flex items-start gap-2">
                  <Badge variant="outline" className="text-xs whitespace-nowrap">
                    {strategy.replace('_', ' ')}
                  </Badge>
                  <p className="text-xs text-muted-foreground">{desc}</p>
                </div>
              ))}
            </CardContent>
          </Card>
        </div>
      </div>
      
      {/* Decision Detail Modal */}
      {showDecisionModal && selectedDecision && (
        <div className="fixed inset-0 bg-black/80 z-50 flex items-center justify-center p-6">
          <div className="max-w-5xl w-full max-h-[90vh] overflow-auto">
            <EnhancedDecisionViewer
              decision={recentDecisions.find(d => d.decisionId === selectedDecision)!}
              onClose={() => setShowDecisionModal(false)}
              isStreaming={false}
            />
          </div>
        </div>
      )}
    </div>
  );
}
