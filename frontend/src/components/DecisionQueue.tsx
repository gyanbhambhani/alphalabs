"use client";

import { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { 
  Clock, 
  TrendingUp, 
  TrendingDown, 
  Minus, 
  Filter,
  Search,
  Loader2,
  CheckCircle2,
  XCircle,
  AlertCircle,
  ArrowRight,
  Activity
} from "lucide-react";
import { DecisionRecord, DecisionStatus } from "@/types";
import { cn } from "@/lib/utils";

interface DecisionQueueProps {
  decisions: DecisionRecord[];
  isLoading?: boolean;
  onDecisionClick?: (decisionId: string) => void;
  showFilters?: boolean;
  enableStreaming?: boolean;
}

// Status to icon mapping
const statusIcons: Record<DecisionStatus, React.ReactNode> = {
  created: <Clock className="h-4 w-4" />,
  debated: <Activity className="h-4 w-4" />,
  risk_vetoed: <XCircle className="h-4 w-4" />,
  finalized: <CheckCircle2 className="h-4 w-4" />,
  sent_to_broker: <Loader2 className="h-4 w-4 animate-spin" />,
  partially_filled: <Loader2 className="h-4 w-4 animate-spin" />,
  filled: <CheckCircle2 className="h-4 w-4" />,
  canceled: <XCircle className="h-4 w-4" />,
  errored: <AlertCircle className="h-4 w-4" />,
};

// Status colors with better visual hierarchy
const statusConfig: Record<DecisionStatus, { 
  color: string; 
  bg: string; 
  border: string;
  label: string;
}> = {
  created: { 
    color: "text-slate-400", 
    bg: "bg-slate-950", 
    border: "border-slate-800",
    label: "Created"
  },
  debated: { 
    color: "text-blue-400", 
    bg: "bg-blue-950/50", 
    border: "border-blue-800",
    label: "Debating"
  },
  risk_vetoed: { 
    color: "text-red-400", 
    bg: "bg-red-950/50", 
    border: "border-red-800",
    label: "Risk Vetoed"
  },
  finalized: { 
    color: "text-emerald-400", 
    bg: "bg-emerald-950/50", 
    border: "border-emerald-800",
    label: "Finalized"
  },
  sent_to_broker: { 
    color: "text-amber-400", 
    bg: "bg-amber-950/50", 
    border: "border-amber-800",
    label: "Sent to Broker"
  },
  partially_filled: { 
    color: "text-orange-400", 
    bg: "bg-orange-950/50", 
    border: "border-orange-800",
    label: "Partially Filled"
  },
  filled: { 
    color: "text-green-400", 
    bg: "bg-green-950/50", 
    border: "border-green-800",
    label: "Filled"
  },
  canceled: { 
    color: "text-gray-400", 
    bg: "bg-gray-950/50", 
    border: "border-gray-800",
    label: "Canceled"
  },
  errored: { 
    color: "text-red-400", 
    bg: "bg-red-950/50", 
    border: "border-red-800",
    label: "Error"
  },
};

function formatDate(dateString: string): string {
  const date = new Date(dateString);
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  const diffMins = Math.floor(diffMs / 60000);
  
  if (diffMins < 1) return "just now";
  if (diffMins < 60) return `${diffMins}m ago`;
  
  const diffHours = Math.floor(diffMins / 60);
  if (diffHours < 24) return `${diffHours}h ago`;
  
  const diffDays = Math.floor(diffHours / 24);
  if (diffDays < 7) return `${diffDays}d ago`;
  
  return date.toLocaleDateString();
}

// Decision Pipeline Visualization
function DecisionPipeline({ status }: { status: DecisionStatus }) {
  const stages = [
    { key: 'created', label: 'Create' },
    { key: 'debated', label: 'Debate' },
    { key: 'finalized', label: 'Finalize' },
    { key: 'sent_to_broker', label: 'Send' },
    { key: 'filled', label: 'Execute' },
  ];
  
  const statusOrder: DecisionStatus[] = [
    'created', 'debated', 'finalized', 'sent_to_broker', 
    'partially_filled', 'filled'
  ];
  
  const currentIndex = statusOrder.indexOf(status);
  
  // Special handling for terminal states
  const isVetoed = status === 'risk_vetoed';
  const isCanceled = status === 'canceled';
  const isErrored = status === 'errored';
  
  return (
    <div className="flex items-center gap-1 mt-2">
      {stages.map((stage, idx) => {
        const stageIndex = statusOrder.indexOf(stage.key as DecisionStatus);
        let stageStatus: 'complete' | 'current' | 'pending' | 'error' = 'pending';
        
        if (isVetoed || isCanceled || isErrored) {
          stageStatus = stageIndex < currentIndex ? 'complete' : 'error';
        } else if (stageIndex < currentIndex) {
          stageStatus = 'complete';
        } else if (stageIndex === currentIndex || 
                   (status === 'partially_filled' && stage.key === 'sent_to_broker')) {
          stageStatus = 'current';
        }
        
        return (
          <div key={stage.key} className="flex items-center">
            <div className="flex flex-col items-center">
              <div className={cn(
                "h-2 w-2 rounded-full transition-all",
                stageStatus === 'complete' && "bg-emerald-500",
                stageStatus === 'current' && "bg-blue-500 animate-pulse",
                stageStatus === 'pending' && "bg-slate-700",
                stageStatus === 'error' && "bg-red-500"
              )} />
              <span className={cn(
                "text-[9px] mt-0.5 uppercase tracking-wide",
                stageStatus === 'complete' && "text-emerald-400",
                stageStatus === 'current' && "text-blue-400 font-medium",
                stageStatus === 'pending' && "text-slate-600",
                stageStatus === 'error' && "text-red-400"
              )}>
                {stage.label}
              </span>
            </div>
            {idx < stages.length - 1 && (
              <div className={cn(
                "h-0.5 w-4 mx-0.5 transition-all",
                stageIndex < currentIndex ? "bg-emerald-500" : "bg-slate-800"
              )} />
            )}
          </div>
        );
      })}
    </div>
  );
}

export function DecisionQueue({ 
  decisions, 
  isLoading = false,
  onDecisionClick,
  showFilters = true,
  enableStreaming = false
}: DecisionQueueProps) {
  const [filterType, setFilterType] = useState<'all' | 'trade' | 'no_trade'>('all');
  const [filterStatus, setFilterStatus] = useState<DecisionStatus | 'all'>('all');
  const [searchQuery, setSearchQuery] = useState('');
  const [showFiltersPanel, setShowFiltersPanel] = useState(false);

  // Filter decisions
  const filteredDecisions = decisions.filter(decision => {
    const typeMatch = filterType === 'all' || decision.decisionType === filterType;
    const statusMatch = filterStatus === 'all' || decision.status === filterStatus;
    const searchMatch = searchQuery === '' || 
      decision.fundId.toLowerCase().includes(searchQuery.toLowerCase()) ||
      Object.keys(decision.predictedDirections || {}).some(symbol => 
        symbol.toLowerCase().includes(searchQuery.toLowerCase())
      );
    
    return typeMatch && statusMatch && searchMatch;
  });

  // Group by status for better organization
  const activeDecisions = filteredDecisions.filter(d => 
    ['created', 'debated', 'sent_to_broker', 'partially_filled'].includes(d.status)
  );
  const completedDecisions = filteredDecisions.filter(d => 
    ['filled', 'finalized'].includes(d.status)
  );
  const failedDecisions = filteredDecisions.filter(d => 
    ['risk_vetoed', 'canceled', 'errored'].includes(d.status)
  );

  return (
    <Card className="bg-zinc-900 border-zinc-800">
      <CardHeader className="pb-3 border-b border-zinc-800">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <CardTitle className="text-sm font-medium text-zinc-300">
              Decision Queue
            </CardTitle>
            {enableStreaming && (
              <div className="flex items-center gap-1.5">
                <div className="h-1.5 w-1.5 rounded-full bg-green-500 animate-pulse" />
                <span className="text-xs text-zinc-500">Live</span>
              </div>
            )}
            {isLoading && (
              <Loader2 className="h-4 w-4 text-zinc-500 animate-spin" />
            )}
          </div>
          
          <div className="flex items-center gap-2">
            <span className="text-xs text-zinc-500">
              {filteredDecisions.length} decision{filteredDecisions.length !== 1 ? 's' : ''}
            </span>
            {showFilters && (
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setShowFiltersPanel(!showFiltersPanel)}
                className="h-7 px-2 text-zinc-400 hover:text-white"
              >
                <Filter className="h-3.5 w-3.5" />
              </Button>
            )}
          </div>
        </div>
        
        {/* Filters Panel */}
        {showFilters && showFiltersPanel && (
          <div className="mt-3 pt-3 border-t border-zinc-800 space-y-3">
            {/* Search */}
            <div className="relative">
              <Search className="absolute left-2.5 top-2.5 h-4 w-4 text-zinc-500" />
              <Input
                placeholder="Search by fund or symbol..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="pl-9 h-9 bg-zinc-800 border-zinc-700 text-sm"
              />
            </div>
            
            {/* Quick Filters */}
            <div className="flex items-center gap-2">
              <span className="text-xs text-zinc-500">Type:</span>
              {(['all', 'trade', 'no_trade'] as const).map(type => (
                <Button
                  key={type}
                  variant="ghost"
                  size="sm"
                  onClick={() => setFilterType(type)}
                  className={cn(
                    "h-7 px-3 text-xs",
                    filterType === type 
                      ? "bg-zinc-700 text-white" 
                      : "text-zinc-500 hover:text-white"
                  )}
                >
                  {type === 'all' ? 'All' : type === 'trade' ? 'Trades' : 'No Trade'}
                </Button>
              ))}
            </div>
          </div>
        )}
      </CardHeader>

      <CardContent className="p-3 space-y-4">
        {isLoading && decisions.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-12 text-zinc-500">
            <Loader2 className="h-8 w-8 animate-spin mb-3" />
            <p className="text-sm">Loading decisions...</p>
          </div>
        ) : filteredDecisions.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-12 text-zinc-500">
            <p className="text-sm">No decisions found</p>
            {(filterType !== 'all' || filterStatus !== 'all' || searchQuery) && (
              <Button
                variant="ghost"
                size="sm"
                onClick={() => {
                  setFilterType('all');
                  setFilterStatus('all');
                  setSearchQuery('');
                }}
                className="mt-2 text-xs text-zinc-400 hover:text-white"
              >
                Clear filters
              </Button>
            )}
          </div>
        ) : (
          <>
            {/* Active Decisions */}
            {activeDecisions.length > 0 && (
              <div className="space-y-2">
                <h4 className="text-xs text-zinc-500 uppercase tracking-wide px-1">
                  Active ({activeDecisions.length})
                </h4>
                <div className="space-y-2">
                  {activeDecisions.map(decision => (
                    <DecisionCard 
                      key={decision.decisionId} 
                      decision={decision} 
                      onClick={() => onDecisionClick?.(decision.decisionId)}
                    />
                  ))}
                </div>
              </div>
            )}
            
            {/* Completed Decisions */}
            {completedDecisions.length > 0 && (
              <div className="space-y-2">
                <h4 className="text-xs text-zinc-500 uppercase tracking-wide px-1">
                  Completed ({completedDecisions.length})
                </h4>
                <div className="space-y-2">
                  {completedDecisions.slice(0, 5).map(decision => (
                    <DecisionCard 
                      key={decision.decisionId} 
                      decision={decision} 
                      onClick={() => onDecisionClick?.(decision.decisionId)}
                    />
                  ))}
                </div>
              </div>
            )}
            
            {/* Failed Decisions */}
            {failedDecisions.length > 0 && (
              <div className="space-y-2">
                <h4 className="text-xs text-zinc-500 uppercase tracking-wide px-1">
                  Failed ({failedDecisions.length})
                </h4>
                <div className="space-y-2">
                  {failedDecisions.slice(0, 3).map(decision => (
                    <DecisionCard 
                      key={decision.decisionId} 
                      decision={decision} 
                      onClick={() => onDecisionClick?.(decision.decisionId)}
                    />
                  ))}
                </div>
              </div>
            )}
          </>
        )}
      </CardContent>
    </Card>
  );
}

// Individual Decision Card
function DecisionCard({ 
  decision, 
  onClick 
}: { 
  decision: DecisionRecord; 
  onClick?: () => void;
}) {
  const config = statusConfig[decision.status];
  const icon = statusIcons[decision.status];
  
  const symbols = Object.keys(decision.predictedDirections || {});
  const directions = Object.values(decision.predictedDirections || {});
  
  return (
    <div
      onClick={onClick}
      className={cn(
        "p-3 rounded-lg border transition-all cursor-pointer",
        "hover:bg-zinc-800/50",
        config.bg,
        config.border
      )}
    >
      {/* Header */}
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <div className={cn("flex items-center gap-1.5", config.color)}>
            {icon}
            <span className="text-xs font-medium uppercase tracking-wide">
              {config.label}
            </span>
          </div>
          <Badge 
            variant={decision.decisionType === 'trade' ? 'default' : 'secondary'}
            className="text-[10px] h-5"
          >
            {decision.decisionType}
          </Badge>
        </div>
        <span className="text-xs text-zinc-500">
          {formatDate(decision.asofTimestamp)}
        </span>
      </div>
      
      {/* Fund & Symbols */}
      <div className="flex items-center justify-between mb-2">
        <span className="text-sm font-medium text-white">
          {decision.fundId}
        </span>
        {symbols.length > 0 && (
          <div className="flex items-center gap-1.5">
            {symbols.slice(0, 3).map((symbol, idx) => {
              const direction = directions[idx];
              const DirectionIcon = direction === 'up' 
                ? TrendingUp 
                : direction === 'down' 
                ? TrendingDown 
                : Minus;
              const colorClass = direction === 'up' 
                ? 'text-green-400' 
                : direction === 'down' 
                ? 'text-red-400' 
                : 'text-zinc-500';
              
              return (
                <div key={symbol} className="flex items-center gap-1">
                  <span className="text-xs font-mono font-bold text-white">
                    {symbol}
                  </span>
                  <DirectionIcon className={cn("h-3 w-3", colorClass)} />
                </div>
              );
            })}
            {symbols.length > 3 && (
              <span className="text-xs text-zinc-500">
                +{symbols.length - 3}
              </span>
            )}
          </div>
        )}
      </div>
      
      {/* No Trade Reason */}
      {decision.noTradeReason && (
        <p className="text-xs text-zinc-400 mb-2">
          {decision.noTradeReason}
        </p>
      )}
      
      {/* Expected Outcomes */}
      {decision.decisionType === 'trade' && (
        <div className="flex items-center gap-4 text-xs text-zinc-500 mb-2">
          {decision.expectedReturn !== undefined && (
            <span>
              Return: <span className="text-zinc-300">
                {(decision.expectedReturn * 100).toFixed(1)}%
              </span>
            </span>
          )}
          {decision.expectedHoldingDays !== undefined && (
            <span>
              Hold: <span className="text-zinc-300">
                {decision.expectedHoldingDays}d
              </span>
            </span>
          )}
        </div>
      )}
      
      {/* Pipeline Visualization */}
      {decision.decisionType === 'trade' && (
        <DecisionPipeline status={decision.status} />
      )}
      
      {/* Click indicator */}
      <div className="flex items-center justify-end mt-2">
        <span className="text-xs text-zinc-600 flex items-center gap-1">
          View details
          <ArrowRight className="h-3 w-3" />
        </span>
      </div>
    </div>
  );
}
