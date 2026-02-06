"use client";

import { useState, useEffect } from "react";
import {
  DebateTranscript,
  DebateMessage,
  DecisionRecord,
  DecisionStatus,
} from "@/types";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { 
  CheckCircle2, 
  XCircle, 
  AlertCircle, 
  Clock, 
  TrendingUp,
  TrendingDown,
  Activity,
  Loader2,
  ArrowRight,
  MessageSquare,
  BarChart3,
  FileText,
  History
} from "lucide-react";
import { cn } from "@/lib/utils";

interface EnhancedDecisionViewerProps {
  decision: DecisionRecord;
  transcript?: DebateTranscript;
  consensusBoard?: ConsensusBoard;
  onClose?: () => void;
  isStreaming?: boolean;
}

const statusConfig: Record<DecisionStatus, {
  icon: React.ReactNode;
  color: string;
  bg: string;
  label: string;
  description: string;
}> = {
  created: {
    icon: <Clock className="h-5 w-5" />,
    color: "text-slate-400",
    bg: "bg-slate-950",
    label: "Created",
    description: "Decision has been created and queued"
  },
  debated: {
    icon: <Activity className="h-5 w-5" />,
    color: "text-blue-400",
    bg: "bg-blue-950/50",
    label: "In Debate",
    description: "AI agents are debating this decision"
  },
  risk_vetoed: {
    icon: <XCircle className="h-5 w-5" />,
    color: "text-red-400",
    bg: "bg-red-950/50",
    label: "Risk Vetoed",
    description: "Decision failed risk management checks"
  },
  finalized: {
    icon: <CheckCircle2 className="h-5 w-5" />,
    color: "text-emerald-400",
    bg: "bg-emerald-950/50",
    label: "Finalized",
    description: "Decision approved and ready for execution"
  },
  sent_to_broker: {
    icon: <Loader2 className="h-5 w-5 animate-spin" />,
    color: "text-amber-400",
    bg: "bg-amber-950/50",
    label: "Sent to Broker",
    description: "Order sent to broker, awaiting execution"
  },
  partially_filled: {
    icon: <Loader2 className="h-5 w-5 animate-spin" />,
    color: "text-orange-400",
    bg: "bg-orange-950/50",
    label: "Partially Filled",
    description: "Order is being filled incrementally"
  },
  filled: {
    icon: <CheckCircle2 className="h-5 w-5" />,
    color: "text-green-400",
    bg: "bg-green-950/50",
    label: "Filled",
    description: "Order completely executed"
  },
  canceled: {
    icon: <XCircle className="h-5 w-5" />,
    color: "text-gray-400",
    bg: "bg-gray-950/50",
    label: "Canceled",
    description: "Order was canceled before execution"
  },
  errored: {
    icon: <AlertCircle className="h-5 w-5" />,
    color: "text-red-400",
    bg: "bg-red-950/50",
    label: "Error",
    description: "An error occurred during processing"
  },
};

function formatDate(dateString: string): string {
  return new Date(dateString).toLocaleString();
}

function formatTime(dateString: string): string {
  return new Date(dateString).toLocaleTimeString();
}

// Execution Timeline Component
function ExecutionTimeline({ decision }: { decision: DecisionRecord }) {
  const stages = [
    { 
      status: 'created', 
      label: 'Created',
      time: decision.asofTimestamp,
      icon: <Clock className="h-4 w-4" />
    },
    { 
      status: 'debated', 
      label: 'Debated',
      time: decision.asofTimestamp, // Would need actual timestamp from backend
      icon: <MessageSquare className="h-4 w-4" />
    },
    { 
      status: 'finalized', 
      label: 'Finalized',
      time: decision.asofTimestamp,
      icon: <CheckCircle2 className="h-4 w-4" />
    },
    { 
      status: 'sent_to_broker', 
      label: 'Sent to Broker',
      time: decision.asofTimestamp,
      icon: <ArrowRight className="h-4 w-4" />
    },
    { 
      status: 'filled', 
      label: 'Executed',
      time: decision.asofTimestamp,
      icon: <Activity className="h-4 w-4" />
    },
  ];

  const statusOrder: DecisionStatus[] = [
    'created', 'debated', 'finalized', 'sent_to_broker', 
    'partially_filled', 'filled'
  ];
  
  const currentIndex = statusOrder.indexOf(decision.status);
  const isTerminalError = ['risk_vetoed', 'canceled', 'errored'].includes(decision.status);

  return (
    <div className="space-y-3">
      {stages.map((stage, idx) => {
        const stageIndex = statusOrder.indexOf(stage.status as DecisionStatus);
        let stageState: 'complete' | 'current' | 'pending' | 'error' = 'pending';
        
        if (isTerminalError && stageIndex >= currentIndex) {
          stageState = 'error';
        } else if (stageIndex < currentIndex) {
          stageState = 'complete';
        } else if (stageIndex === currentIndex || 
                   (decision.status === 'partially_filled' && stage.status === 'sent_to_broker')) {
          stageState = 'current';
        }

        return (
          <div key={stage.status} className="flex items-start gap-3">
            {/* Timeline dot and line */}
            <div className="flex flex-col items-center">
              <div className={cn(
                "h-8 w-8 rounded-full flex items-center justify-center border-2 transition-all",
                stageState === 'complete' && "bg-emerald-500 border-emerald-500 text-white",
                stageState === 'current' && "bg-blue-500 border-blue-500 text-white animate-pulse",
                stageState === 'pending' && "bg-zinc-900 border-zinc-700 text-zinc-600",
                stageState === 'error' && "bg-red-500 border-red-500 text-white"
              )}>
                {stage.icon}
              </div>
              {idx < stages.length - 1 && (
                <div className={cn(
                  "w-0.5 h-8 transition-all",
                  stageIndex < currentIndex ? "bg-emerald-500" : "bg-zinc-800"
                )} />
              )}
            </div>
            
            {/* Stage info */}
            <div className="flex-1 pt-1">
              <div className="flex items-center justify-between">
                <h4 className={cn(
                  "text-sm font-medium",
                  stageState === 'complete' && "text-emerald-400",
                  stageState === 'current' && "text-blue-400",
                  stageState === 'pending' && "text-zinc-600",
                  stageState === 'error' && "text-red-400"
                )}>
                  {stage.label}
                </h4>
                {stageState !== 'pending' && (
                  <span className="text-xs text-zinc-500">
                    {formatTime(stage.time)}
                  </span>
                )}
              </div>
              {stageState === 'current' && (
                <p className="text-xs text-zinc-500 mt-1">
                  {statusConfig[decision.status].description}
                </p>
              )}
            </div>
          </div>
        );
      })}
      
      {/* Error state */}
      {isTerminalError && (
        <div className="flex items-start gap-3">
          <div className="h-8 w-8 rounded-full flex items-center justify-center border-2 bg-red-500 border-red-500 text-white">
            <XCircle className="h-4 w-4" />
          </div>
          <div className="flex-1 pt-1">
            <h4 className="text-sm font-medium text-red-400">
              {statusConfig[decision.status].label}
            </h4>
            <p className="text-xs text-zinc-500 mt-1">
              {statusConfig[decision.status].description}
            </p>
            {decision.noTradeReason && (
              <p className="text-xs text-red-400 mt-1">
                Reason: {decision.noTradeReason}
              </p>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

// Consensus Gate Status Types
interface ConsensusGate {
  action_match: boolean;
  symbol_match: boolean;
  thesis_type_match: boolean;
  horizon_within_tolerance: boolean;
  invalidation_compatible: boolean;
  min_confidence_met: boolean;
  evidence_validated: boolean;
  min_edge_met: boolean;
  is_executable: boolean;
}

interface ConsensusBoard {
  gate: ConsensusGate;
  agreed_action: string | null;
  agreed_symbol: string | null;
  agreed_thesis_type: string | null;
  agreed_horizon: number | null;
  alignment_score: number;
  open_questions: string[];
  concessions_made: string[];
  disputed_topics: string[];
  dispute_summary: string;
  can_execute: boolean;
}

// Consensus Board Widget
function ConsensusBoardWidget({ 
  consensusBoard 
}: { 
  consensusBoard?: ConsensusBoard 
}) {
  if (!consensusBoard) {
    return (
      <Card className="bg-zinc-900 border-zinc-800">
        <CardContent className="p-4 text-center text-zinc-500">
          <p className="text-sm">No consensus data available</p>
        </CardContent>
      </Card>
    );
  }

  const gate = consensusBoard.gate;
  const passedCount = Object.values(gate).filter(v => v === true).length - 1; // -1 for is_executable
  const totalGates = 8;

  return (
    <Card className="bg-zinc-900 border-zinc-800">
      <CardHeader className="pb-3">
        <CardTitle className="text-sm font-medium text-zinc-300 uppercase tracking-wide flex items-center gap-2">
          Consensus Board
          <Badge 
            variant={consensusBoard.can_execute ? "default" : "secondary"}
            className={cn(
              consensusBoard.can_execute 
                ? "bg-emerald-500/20 text-emerald-400" 
                : "bg-zinc-700 text-zinc-400"
            )}
          >
            {consensusBoard.can_execute ? "EXECUTABLE" : "NOT READY"}
          </Badge>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Agreed Items */}
        <div className="space-y-2">
          <p className="text-xs text-zinc-500 uppercase tracking-wide">Agreed</p>
          <div className="space-y-1">
            {consensusBoard.agreed_action && (
              <div className="flex items-center gap-2">
                <CheckCircle2 className="h-4 w-4 text-emerald-400" />
                <span className="text-sm text-zinc-300">
                  Action: <span className="font-medium text-white uppercase">
                    {consensusBoard.agreed_action}
                  </span>
                </span>
              </div>
            )}
            {consensusBoard.agreed_symbol && (
              <div className="flex items-center gap-2">
                <CheckCircle2 className="h-4 w-4 text-emerald-400" />
                <span className="text-sm text-zinc-300">
                  Symbol: <span className="font-mono font-medium text-white">
                    {consensusBoard.agreed_symbol}
                  </span>
                </span>
              </div>
            )}
            {consensusBoard.agreed_thesis_type && (
              <div className="flex items-center gap-2">
                <CheckCircle2 className="h-4 w-4 text-emerald-400" />
                <span className="text-sm text-zinc-300">
                  Thesis: <span className="font-medium text-white">
                    {consensusBoard.agreed_thesis_type}
                  </span>
                </span>
              </div>
            )}
            {consensusBoard.agreed_horizon && (
              <div className="flex items-center gap-2">
                <CheckCircle2 className="h-4 w-4 text-emerald-400" />
                <span className="text-sm text-zinc-300">
                  Horizon: <span className="font-medium text-white">
                    {consensusBoard.agreed_horizon}d
                  </span>
                </span>
              </div>
            )}
          </div>
        </div>

        {/* Disputed Topics */}
        {consensusBoard.disputed_topics.length > 0 && (
          <div className="space-y-2">
            <p className="text-xs text-zinc-500 uppercase tracking-wide">Disputed</p>
            <div className="space-y-1">
              {consensusBoard.disputed_topics.map((topic, idx) => (
                <div key={idx} className="flex items-center gap-2">
                  <AlertCircle className="h-4 w-4 text-amber-400" />
                  <span className="text-sm text-zinc-400">{topic}</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Open Questions */}
        {consensusBoard.open_questions.length > 0 && (
          <div className="space-y-2">
            <p className="text-xs text-zinc-500 uppercase tracking-wide">Open Questions</p>
            <div className="space-y-1">
              {consensusBoard.open_questions.map((q, idx) => (
                <div key={idx} className="flex items-start gap-2">
                  <span className="text-amber-400">?</span>
                  <span className="text-sm text-zinc-400">{q}</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Gate Status */}
        <div className="space-y-2">
          <p className="text-xs text-zinc-500 uppercase tracking-wide">
            Gate Status ({passedCount}/{totalGates})
          </p>
          <div className="grid grid-cols-2 gap-2">
            {[
              { key: 'action_match', label: 'Action' },
              { key: 'symbol_match', label: 'Symbol' },
              { key: 'thesis_type_match', label: 'Thesis' },
              { key: 'horizon_within_tolerance', label: 'Horizon' },
              { key: 'invalidation_compatible', label: 'Invalidation' },
              { key: 'min_confidence_met', label: 'Confidence' },
              { key: 'evidence_validated', label: 'Evidence' },
              { key: 'min_edge_met', label: 'Edge' },
            ].map(({ key, label }) => {
              const passed = gate[key as keyof ConsensusGate];
              return (
                <div 
                  key={key}
                  className={cn(
                    "flex items-center gap-2 text-xs px-2 py-1 rounded",
                    passed ? "bg-emerald-500/10" : "bg-zinc-800"
                  )}
                >
                  {passed ? (
                    <CheckCircle2 className="h-3 w-3 text-emerald-400" />
                  ) : (
                    <XCircle className="h-3 w-3 text-zinc-500" />
                  )}
                  <span className={passed ? "text-emerald-400" : "text-zinc-500"}>
                    {label}
                  </span>
                </div>
              );
            })}
          </div>
        </div>

        {/* Alignment Score */}
        <div className="pt-2 border-t border-zinc-800">
          <div className="flex items-center justify-between">
            <span className="text-xs text-zinc-500 uppercase tracking-wide">
              Alignment
            </span>
            <span className="text-lg font-bold text-white">
              {(consensusBoard.alignment_score * 100).toFixed(0)}%
            </span>
          </div>
          <div className="mt-2 h-2 bg-zinc-800 rounded-full overflow-hidden">
            <div 
              className={cn(
                "h-full rounded-full transition-all",
                consensusBoard.alignment_score >= 0.7 
                  ? "bg-emerald-500" 
                  : consensusBoard.alignment_score >= 0.5 
                  ? "bg-amber-500" 
                  : "bg-red-500"
              )}
              style={{ width: `${consensusBoard.alignment_score * 100}%` }}
            />
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

// Three-Lane Turn Display
interface AgentTurnData {
  agent_id: string;
  round_num: number;
  action: string;
  symbol: string | null;
  suggested_weight: number;
  risk_posture: string;
  thesis: {
    thesis_type: string;
    horizon_days: number;
    primary_signal: string;
    secondary_signal: string | null;
  };
  confidence: {
    signal_strength: number;
    regime_fit: number;
    risk_comfort: number;
    execution_feasibility: number;
  };
  evidence_cited: Array<{ feature: string; symbol: string }>;
  dialog_move: {
    acknowledge: string;
    challenge: string | null;
    request: string | null;
    concede_or_hold: string;
  };
  counterfactual: {
    alternative_action: string;
    why_rejected: string;
  };
}

function ThreeLaneTurnDisplay({ turn }: { turn: AgentTurnData }) {
  const overallConfidence = Math.pow(
    turn.confidence.signal_strength *
    turn.confidence.regime_fit *
    turn.confidence.risk_comfort *
    turn.confidence.execution_feasibility,
    0.25
  );

  return (
    <Card className="bg-zinc-900 border-zinc-800">
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Badge 
              variant="outline" 
              className={cn(
                turn.agent_id === 'analyst' 
                  ? "border-blue-500 text-blue-400" 
                  : "border-purple-500 text-purple-400"
              )}
            >
              {turn.agent_id.toUpperCase()}
            </Badge>
            <span className="text-xs text-zinc-500">Round {turn.round_num + 1}</span>
          </div>
          <Badge 
            className={cn(
              "uppercase",
              turn.action === 'buy' && "bg-emerald-500/20 text-emerald-400",
              turn.action === 'sell' && "bg-red-500/20 text-red-400",
              turn.action === 'hold' && "bg-zinc-700 text-zinc-400"
            )}
          >
            {turn.action}
          </Badge>
        </div>
      </CardHeader>
      <CardContent className="pt-2">
        <div className="grid grid-cols-3 gap-4">
          {/* Claim Lane */}
          <div className="space-y-2">
            <p className="text-xs text-zinc-500 uppercase tracking-wide">Claim</p>
            <div className="bg-zinc-950 rounded-lg p-3 text-sm">
              <p className="text-zinc-300">
                <span className="font-medium text-white">{turn.thesis.thesis_type}</span>
                {" on "}
                <span className="font-mono text-white">{turn.symbol || "N/A"}</span>
              </p>
              <p className="text-zinc-500 text-xs mt-1">
                {turn.thesis.horizon_days}d horizon
              </p>
              {turn.dialog_move.acknowledge && (
                <p className="text-zinc-400 text-xs mt-2 italic">
                  "{turn.dialog_move.acknowledge.slice(0, 100)}..."
                </p>
              )}
            </div>
          </div>

          {/* Evidence Lane */}
          <div className="space-y-2">
            <p className="text-xs text-zinc-500 uppercase tracking-wide">Evidence</p>
            <div className="bg-zinc-950 rounded-lg p-3 text-sm space-y-1">
              {turn.evidence_cited.slice(0, 4).map((ev, idx) => (
                <div key={idx} className="flex items-center gap-2 text-xs">
                  <span className="text-zinc-500">•</span>
                  <span className="text-zinc-300">
                    {ev.feature}
                  </span>
                  <span className="text-zinc-500 font-mono">
                    ({ev.symbol})
                  </span>
                </div>
              ))}
              {turn.evidence_cited.length > 4 && (
                <p className="text-xs text-zinc-500">
                  +{turn.evidence_cited.length - 4} more
                </p>
              )}
            </div>
          </div>

          {/* Action Lane */}
          <div className="space-y-2">
            <p className="text-xs text-zinc-500 uppercase tracking-wide">Action</p>
            <div className="bg-zinc-950 rounded-lg p-3 text-sm space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-zinc-500">Weight</span>
                <span className="font-medium text-white">
                  {(turn.suggested_weight * 100).toFixed(0)}%
                </span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-zinc-500">Confidence</span>
                <span className="font-medium text-white">
                  {(overallConfidence * 100).toFixed(0)}%
                </span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-zinc-500">Risk</span>
                <Badge variant="outline" className="text-xs">
                  {turn.risk_posture}
                </Badge>
              </div>
            </div>
          </div>
        </div>

        {/* Dialog Moves */}
        {(turn.dialog_move.challenge || turn.dialog_move.request) && (
          <div className="mt-3 pt-3 border-t border-zinc-800 space-y-2">
            {turn.dialog_move.challenge && (
              <div className="flex items-start gap-2 text-xs">
                <span className="text-amber-400 font-medium">CHALLENGE:</span>
                <span className="text-zinc-400">{turn.dialog_move.challenge}</span>
              </div>
            )}
            {turn.dialog_move.request && (
              <div className="flex items-start gap-2 text-xs">
                <span className="text-blue-400 font-medium">REQUEST:</span>
                <span className="text-zinc-400">{turn.dialog_move.request}</span>
              </div>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
}

// Enhanced Debate Viewer
function EnhancedDebateView({ 
  transcript, 
  isStreaming,
  consensusBoard 
}: { 
  transcript?: DebateTranscript;
  isStreaming?: boolean;
  consensusBoard?: ConsensusBoard;
}) {
  const [selectedPhase, setSelectedPhase] = useState<string | null>(null);
  const [viewMode, setViewMode] = useState<'lanes' | 'raw'>('lanes');

  if (!transcript) {
    return (
      <div className="flex flex-col items-center justify-center py-12 text-zinc-500">
        {isStreaming ? (
          <>
            <Loader2 className="h-8 w-8 animate-spin mb-3" />
            <p className="text-sm">Debate in progress...</p>
          </>
        ) : (
          <p className="text-sm">No debate transcript available</p>
        )}
      </div>
    );
  }

  // Group messages by phase
  const messagesByPhase = transcript.messages?.reduce((acc, msg) => {
    if (!acc[msg.phase]) {
      acc[msg.phase] = [];
    }
    acc[msg.phase].push(msg);
    return acc;
  }, {} as Record<string, DebateMessage[]>) || {};

  const phases = Object.keys(messagesByPhase);
  
  // Try to parse agent turns from messages
  const agentTurns: AgentTurnData[] = [];
  transcript.messages?.forEach(msg => {
    try {
      const content = typeof msg.content === 'string' 
        ? msg.content 
        : (msg.content as any)?.text || JSON.stringify(msg.content);
      
      // Try to extract JSON from content - handle nested braces
      let jsonStr = content;
      
      // If content starts with text before JSON, find the JSON object
      const jsonStart = content.indexOf('{');
      if (jsonStart > 0) {
        jsonStr = content.slice(jsonStart);
      }
      
      // Find matching closing brace
      let braceCount = 0;
      let jsonEnd = -1;
      for (let i = 0; i < jsonStr.length; i++) {
        if (jsonStr[i] === '{') braceCount++;
        if (jsonStr[i] === '}') braceCount--;
        if (braceCount === 0 && jsonStr[i] === '}') {
          jsonEnd = i + 1;
          break;
        }
      }
      
      if (jsonEnd > 0) {
        const parsed = JSON.parse(jsonStr.slice(0, jsonEnd));
        
        // Check for V2 structure (action + thesis) or V1 structure
        if (parsed.action && parsed.thesis) {
          // Extract agent_id from phase name (e.g., "round_0_analyst")
          const phaseMatch = msg.phase?.match(/round_(\d+)_(\w+)/);
          const agentId = phaseMatch ? phaseMatch[2] : (msg.participantId || 'unknown');
          const roundNum = phaseMatch ? parseInt(phaseMatch[1]) : 0;
          
          agentTurns.push({
            agent_id: agentId,
            round_num: roundNum,
            action: parsed.action || 'hold',
            symbol: parsed.symbol || null,
            suggested_weight: parsed.suggested_weight || 0,
            risk_posture: parsed.risk_posture || 'normal',
            thesis: parsed.thesis || { thesis_type: 'momentum', horizon_days: 5, primary_signal: 'return_1d', secondary_signal: null },
            confidence: parsed.confidence || { signal_strength: 0.5, regime_fit: 0.5, risk_comfort: 0.5, execution_feasibility: 0.5 },
            evidence_cited: parsed.evidence_cited || [],
            dialog_move: parsed.dialog_move || { acknowledge: '', challenge: null, request: null, concede_or_hold: '' },
            counterfactual: parsed.counterfactual || { alternative_action: 'hold', why_rejected: '' },
          });
        }
      }
    } catch (e) {
      // Not a structured turn, skip
      console.debug('Failed to parse agent turn:', e);
    }
  });
  
  const phaseConfig: Record<string, { icon: React.ReactNode; color: string; label: string }> = {
    propose: { 
      icon: <TrendingUp className="h-4 w-4" />, 
      color: "text-blue-400", 
      label: "Proposals" 
    },
    critique: { 
      icon: <MessageSquare className="h-4 w-4" />, 
      color: "text-amber-400", 
      label: "Critiques" 
    },
    synthesize: { 
      icon: <BarChart3 className="h-4 w-4" />, 
      color: "text-purple-400", 
      label: "Synthesis" 
    },
    risk_check: { 
      icon: <AlertCircle className="h-4 w-4" />, 
      color: "text-red-400", 
      label: "Risk Check" 
    },
    finalize: { 
      icon: <CheckCircle2 className="h-4 w-4" />, 
      color: "text-green-400", 
      label: "Final Decision" 
    },
  };

  return (
    <div className="space-y-4">
      {/* Consensus Board Widget */}
      {consensusBoard && <ConsensusBoardWidget consensusBoard={consensusBoard} />}

      {/* Debate Stats */}
      <div className="grid grid-cols-3 gap-4">
        <Card className="bg-zinc-900 border-zinc-800">
          <CardContent className="p-4 text-center">
            <p className="text-2xl font-bold text-blue-400">
              {transcript.numProposals}
            </p>
            <p className="text-xs text-zinc-500 uppercase tracking-wide mt-1">
              Proposals
            </p>
          </CardContent>
        </Card>
        <Card className="bg-zinc-900 border-zinc-800">
          <CardContent className="p-4 text-center">
            <p className="text-2xl font-bold text-amber-400">
              {transcript.numCritiques}
            </p>
            <p className="text-xs text-zinc-500 uppercase tracking-wide mt-1">
              Critiques
            </p>
          </CardContent>
        </Card>
        <Card className="bg-zinc-900 border-zinc-800">
          <CardContent className="p-4 text-center">
            <p className="text-2xl font-bold text-green-400">
              {(transcript.finalConsensusLevel * 100).toFixed(0)}%
            </p>
            <p className="text-xs text-zinc-500 uppercase tracking-wide mt-1">
              Consensus
            </p>
          </CardContent>
        </Card>
      </div>

      {/* View Mode Toggle */}
      <div className="flex items-center gap-2">
        <Button
          variant={viewMode === 'lanes' ? "default" : "outline"}
          size="sm"
          onClick={() => setViewMode('lanes')}
        >
          Three-Lane View
        </Button>
        <Button
          variant={viewMode === 'raw' ? "default" : "outline"}
          size="sm"
          onClick={() => setViewMode('raw')}
        >
          Raw Messages
        </Button>
      </div>

      {/* Three-Lane View */}
      {viewMode === 'lanes' && agentTurns.length > 0 && (
        <div className="space-y-4">
          {agentTurns.map((turn, idx) => (
            <ThreeLaneTurnDisplay key={idx} turn={turn} />
          ))}
        </div>
      )}

      {/* Raw Message View */}
      {viewMode === 'raw' && (
        <>
          {/* Phase Selector */}
          <div className="flex flex-wrap gap-2">
            {phases.map((phase) => {
              const config = phaseConfig[phase] || { 
                icon: <Activity className="h-4 w-4" />, 
                color: "text-zinc-400",
                label: phase 
              };
              return (
                <Button
                  key={phase}
                  variant={selectedPhase === phase ? "default" : "outline"}
                  size="sm"
                  onClick={() => setSelectedPhase(selectedPhase === phase ? null : phase)}
                  className={cn(
                    "h-9",
                    selectedPhase === phase && "bg-zinc-700"
                  )}
                >
                  <span className={config.color}>{config.icon}</span>
                  <span className="ml-2">
                    {config.label} ({messagesByPhase[phase]?.length || 0})
                  </span>
                </Button>
              );
            })}
          </div>

          {/* Messages */}
          {selectedPhase && messagesByPhase[selectedPhase] && (
            <div className="space-y-3 max-h-[500px] overflow-y-auto">
              {messagesByPhase[selectedPhase].map((msg, idx) => {
                const config = phaseConfig[msg.phase] || { color: "text-zinc-400" };
                return (
                  <Card key={idx} className="bg-zinc-900 border-zinc-800">
                    <CardContent className="p-4">
                      <div className="flex items-center justify-between mb-3">
                        <div className="flex items-center gap-2">
                          <span className={config.color}>
                            {phaseConfig[msg.phase]?.icon}
                          </span>
                          <Badge variant="outline" className="text-xs">
                            {msg.participantId}
                          </Badge>
                        </div>
                        <span className="text-xs text-zinc-500 font-mono">
                          {msg.modelName}
                        </span>
                      </div>
                      <div className="bg-zinc-950 rounded-lg p-3 overflow-x-auto">
                        <pre className="text-xs text-zinc-300 whitespace-pre-wrap">
                          {typeof msg.content === 'string' 
                            ? msg.content 
                            : (msg.content as any)?.text || JSON.stringify(msg.content, null, 2)}
                        </pre>
                      </div>
                    </CardContent>
                  </Card>
                );
              })}
            </div>
          )}
        </>
      )}

      {/* No structured turns available */}
      {viewMode === 'lanes' && agentTurns.length === 0 && (
        <div className="text-center py-8 text-zinc-500">
          <p className="text-sm">No structured agent turns available.</p>
          <p className="text-xs mt-1">Switch to Raw Messages view to see debate content.</p>
        </div>
      )}
    </div>
  );
}

export function EnhancedDecisionViewer({
  decision,
  transcript,
  consensusBoard,
  onClose,
  isStreaming = false,
}: EnhancedDecisionViewerProps) {
  const config = statusConfig[decision.status];

  return (
    <Card className="w-full bg-zinc-900 border-zinc-800">
      <CardHeader className="border-b border-zinc-800">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className={cn(
              "h-12 w-12 rounded-lg flex items-center justify-center",
              config.bg,
              config.color
            )}>
              {config.icon}
            </div>
            <div>
              <CardTitle className="text-lg text-white">
                {decision.fundId}
              </CardTitle>
              <p className="text-sm text-zinc-500">
                {formatDate(decision.asofTimestamp)}
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            {isStreaming && (
              <div className="flex items-center gap-1.5">
                <div className="h-2 w-2 rounded-full bg-green-500 animate-pulse" />
                <span className="text-xs text-zinc-500">Live</span>
              </div>
            )}
            <Badge className={cn(
              "text-sm",
              config.color,
              config.bg
            )}>
              {config.label}
            </Badge>
            <Badge variant={decision.decisionType === "trade" ? "default" : "secondary"}>
              {decision.decisionType}
            </Badge>
            {onClose && (
              <Button variant="ghost" size="sm" onClick={onClose}>
                Close
              </Button>
            )}
          </div>
        </div>
      </CardHeader>

      <CardContent className="p-6">
        <Tabs defaultValue="timeline" className="w-full">
          <TabsList className="grid w-full grid-cols-4 bg-zinc-800">
            <TabsTrigger value="timeline">
              <History className="h-4 w-4 mr-2" />
              Timeline
            </TabsTrigger>
            <TabsTrigger value="summary">
              <FileText className="h-4 w-4 mr-2" />
              Summary
            </TabsTrigger>
            <TabsTrigger value="debate" disabled={!transcript && !isStreaming}>
              <MessageSquare className="h-4 w-4 mr-2" />
              Debate
            </TabsTrigger>
            <TabsTrigger value="audit">
              <BarChart3 className="h-4 w-4 mr-2" />
              Audit
            </TabsTrigger>
          </TabsList>

          {/* Timeline Tab */}
          <TabsContent value="timeline" className="space-y-4 mt-6">
            <div className="bg-zinc-950 rounded-lg p-6">
              <h3 className="text-sm font-medium text-zinc-300 mb-4 uppercase tracking-wide">
                Execution Pipeline
              </h3>
              <ExecutionTimeline decision={decision} />
            </div>
          </TabsContent>

          {/* Summary Tab */}
          <TabsContent value="summary" className="space-y-4 mt-6">
            {/* Decision Info */}
            <Card className="bg-zinc-950 border-zinc-800">
              <CardContent className="p-4 grid grid-cols-2 gap-4">
                <div>
                  <p className="text-xs text-zinc-500 uppercase tracking-wide">Fund</p>
                  <p className="font-medium text-white mt-1">{decision.fundId}</p>
                </div>
                <div>
                  <p className="text-xs text-zinc-500 uppercase tracking-wide">Type</p>
                  <p className="font-medium text-white mt-1">{decision.decisionType}</p>
                </div>
                {decision.noTradeReason && (
                  <div className="col-span-2">
                    <p className="text-xs text-zinc-500 uppercase tracking-wide">
                      No-Trade Reason
                    </p>
                    <p className="font-medium text-white mt-1">{decision.noTradeReason}</p>
                  </div>
                )}
              </CardContent>
            </Card>

            {/* Predictions */}
            {decision.predictedDirections &&
              Object.keys(decision.predictedDirections).length > 0 && (
                <Card className="bg-zinc-950 border-zinc-800">
                  <CardContent className="p-4">
                    <h3 className="text-xs text-zinc-500 uppercase tracking-wide mb-3">
                      Predictions
                    </h3>
                    <div className="grid grid-cols-2 gap-2">
                      {Object.entries(decision.predictedDirections).map(
                        ([symbol, direction]) => {
                          const Icon = direction === "up" 
                            ? TrendingUp 
                            : direction === "down" 
                            ? TrendingDown 
                            : Activity;
                          const colorClass = direction === "up"
                            ? "text-green-400"
                            : direction === "down"
                            ? "text-red-400"
                            : "text-zinc-500";
                          
                          return (
                            <div 
                              key={symbol}
                              className="flex items-center justify-between p-3 bg-zinc-900 rounded-lg"
                            >
                              <span className="font-mono font-bold text-white">
                                {symbol}
                              </span>
                              <div className={cn("flex items-center gap-1.5", colorClass)}>
                                <Icon className="h-4 w-4" />
                                <span className="text-sm uppercase">{direction}</span>
                              </div>
                            </div>
                          );
                        }
                      )}
                    </div>
                  </CardContent>
                </Card>
              )}

            {/* Expected Outcomes */}
            {(decision.expectedReturn || decision.expectedHoldingDays) && (
              <Card className="bg-zinc-950 border-zinc-800">
                <CardContent className="p-4 grid grid-cols-2 gap-4">
                  {decision.expectedReturn && (
                    <div>
                      <p className="text-xs text-zinc-500 uppercase tracking-wide">
                        Expected Return
                      </p>
                      <p className="text-2xl font-bold text-emerald-400 mt-1">
                        {(decision.expectedReturn * 100).toFixed(2)}%
                      </p>
                    </div>
                  )}
                  {decision.expectedHoldingDays && (
                    <div>
                      <p className="text-xs text-zinc-500 uppercase tracking-wide">
                        Expected Holding
                      </p>
                      <p className="text-2xl font-bold text-blue-400 mt-1">
                        {decision.expectedHoldingDays} days
                      </p>
                    </div>
                  )}
                </CardContent>
              </Card>
            )}
          </TabsContent>

          {/* Debate Tab */}
          <TabsContent value="debate" className="mt-6">
            <EnhancedDebateView 
              transcript={transcript} 
              isStreaming={isStreaming}
              consensusBoard={consensusBoard}
            />
          </TabsContent>

          {/* Audit Trail Tab */}
          <TabsContent value="audit" className="space-y-4 mt-6">
            <Card className="bg-zinc-950 border-zinc-800">
              <CardContent className="p-4 grid grid-cols-2 gap-4">
                <div className="col-span-2">
                  <p className="text-xs text-zinc-500 uppercase tracking-wide">
                    Decision ID
                  </p>
                  <p className="font-mono text-sm text-white mt-1 break-all">
                    {decision.decisionId}
                  </p>
                </div>
                <div className="col-span-2">
                  <p className="text-xs text-zinc-500 uppercase tracking-wide">
                    Snapshot ID
                  </p>
                  <p className="font-mono text-sm text-white mt-1 break-all">
                    {decision.snapshotId}
                  </p>
                </div>
                {decision.universeHash && (
                  <div>
                    <p className="text-xs text-zinc-500 uppercase tracking-wide">
                      Universe Hash
                    </p>
                    <p className="font-mono text-sm text-white mt-1 break-all">
                      {decision.universeHash}
                    </p>
                  </div>
                )}
                {decision.inputsHash && (
                  <div>
                    <p className="text-xs text-zinc-500 uppercase tracking-wide">
                      Inputs Hash
                    </p>
                    <p className="font-mono text-sm text-white mt-1 break-all">
                      {decision.inputsHash}
                    </p>
                  </div>
                )}
              </CardContent>
            </Card>

            {/* Token Usage */}
            {transcript && (transcript.totalInputTokens || transcript.totalOutputTokens) && (
              <Card className="bg-zinc-950 border-zinc-800">
                <CardContent className="p-4">
                  <h4 className="text-xs text-zinc-500 uppercase tracking-wide mb-3">
                    Token Usage
                  </h4>
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <p className="text-xs text-zinc-500">Input Tokens</p>
                      <p className="text-xl font-bold text-white mt-1">
                        {transcript.totalInputTokens?.toLocaleString()}
                      </p>
                    </div>
                    <div>
                      <p className="text-xs text-zinc-500">Output Tokens</p>
                      <p className="text-xl font-bold text-white mt-1">
                        {transcript.totalOutputTokens?.toLocaleString()}
                      </p>
                    </div>
                  </div>
                </CardContent>
              </Card>
            )}
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
}
