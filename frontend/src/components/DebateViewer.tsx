"use client";

import { useState } from "react";
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

interface DebateViewerProps {
  decision: DecisionRecord;
  transcript?: DebateTranscript;
  onClose?: () => void;
}

const statusColors: Record<DecisionStatus, string> = {
  created: "bg-gray-500",
  debated: "bg-blue-500",
  risk_vetoed: "bg-red-500",
  finalized: "bg-green-500",
  sent_to_broker: "bg-yellow-500",
  partially_filled: "bg-orange-500",
  filled: "bg-green-600",
  canceled: "bg-gray-600",
  errored: "bg-red-600",
};

const phaseLabels: Record<string, string> = {
  propose: "Propose",
  critique: "Critique",
  synthesize: "Synthesize",
  risk_check: "Risk Check",
  finalize: "Finalize",
};

function formatDate(dateString: string): string {
  return new Date(dateString).toLocaleString();
}

export function DebateViewer({
  decision,
  transcript,
  onClose,
}: DebateViewerProps) {
  const [selectedPhase, setSelectedPhase] = useState<string | null>(null);

  // Group messages by phase
  const messagesByPhase = transcript?.messages?.reduce((acc, msg) => {
    if (!acc[msg.phase]) {
      acc[msg.phase] = [];
    }
    acc[msg.phase].push(msg);
    return acc;
  }, {} as Record<string, DebateMessage[]>) || {};

  const phases = Object.keys(messagesByPhase);

  return (
    <Card className="w-full">
      <CardHeader className="flex flex-row items-center justify-between">
        <div>
          <CardTitle className="text-lg">Decision Details</CardTitle>
          <p className="text-sm text-muted-foreground">
            {formatDate(decision.asofTimestamp)}
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Badge className={`${statusColors[decision.status]} text-white`}>
            {decision.status}
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
      </CardHeader>

      <CardContent>
        <Tabs defaultValue="summary" className="w-full">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="summary">Summary</TabsTrigger>
            <TabsTrigger value="debate" disabled={!transcript}>
              Debate
            </TabsTrigger>
            <TabsTrigger value="audit">Audit Trail</TabsTrigger>
          </TabsList>

          {/* Summary Tab */}
          <TabsContent value="summary" className="space-y-4">
            {/* Decision Info */}
            <div className="grid grid-cols-2 gap-4 p-4 bg-muted rounded-lg">
              <div>
                <p className="text-sm text-muted-foreground">Fund</p>
                <p className="font-medium">{decision.fundId}</p>
              </div>
              <div>
                <p className="text-sm text-muted-foreground">Type</p>
                <p className="font-medium">{decision.decisionType}</p>
              </div>
              {decision.noTradeReason && (
                <div className="col-span-2">
                  <p className="text-sm text-muted-foreground">No-Trade Reason</p>
                  <p className="font-medium">{decision.noTradeReason}</p>
                </div>
              )}
            </div>

            {/* Predictions */}
            {decision.predictedDirections &&
              Object.keys(decision.predictedDirections).length > 0 && (
                <div>
                  <h3 className="font-semibold mb-2">Predictions</h3>
                  <div className="flex flex-wrap gap-2">
                    {Object.entries(decision.predictedDirections).map(
                      ([symbol, direction]) => (
                        <Badge
                          key={symbol}
                          variant={direction === "up" ? "default" : "destructive"}
                        >
                          {symbol}: {direction}
                        </Badge>
                      )
                    )}
                  </div>
                </div>
              )}

            {/* Expected Outcomes */}
            {(decision.expectedReturn || decision.expectedHoldingDays) && (
              <div className="grid grid-cols-2 gap-4">
                {decision.expectedReturn && (
                  <div>
                    <p className="text-sm text-muted-foreground">Expected Return</p>
                    <p className="font-medium">
                      {(decision.expectedReturn * 100).toFixed(2)}%
                    </p>
                  </div>
                )}
                {decision.expectedHoldingDays && (
                  <div>
                    <p className="text-sm text-muted-foreground">
                      Expected Holding
                    </p>
                    <p className="font-medium">
                      {decision.expectedHoldingDays} days
                    </p>
                  </div>
                )}
              </div>
            )}
          </TabsContent>

          {/* Debate Tab - Drill Down */}
          <TabsContent value="debate" className="space-y-4">
            {transcript ? (
              <>
                {/* Debate Stats */}
                <div className="grid grid-cols-3 gap-4 p-4 bg-muted rounded-lg">
                  <div className="text-center">
                    <p className="text-2xl font-bold">{transcript.numProposals}</p>
                    <p className="text-sm text-muted-foreground">Proposals</p>
                  </div>
                  <div className="text-center">
                    <p className="text-2xl font-bold">{transcript.numCritiques}</p>
                    <p className="text-sm text-muted-foreground">Critiques</p>
                  </div>
                  <div className="text-center">
                    <p className="text-2xl font-bold">
                      {(transcript.finalConsensusLevel * 100).toFixed(0)}%
                    </p>
                    <p className="text-sm text-muted-foreground">Consensus</p>
                  </div>
                </div>

                {/* Phase Selector */}
                <div className="flex flex-wrap gap-2">
                  {phases.map((phase) => (
                    <Button
                      key={phase}
                      variant={selectedPhase === phase ? "default" : "outline"}
                      size="sm"
                      onClick={() =>
                        setSelectedPhase(selectedPhase === phase ? null : phase)
                      }
                    >
                      {phaseLabels[phase] || phase} (
                      {messagesByPhase[phase]?.length || 0})
                    </Button>
                  ))}
                </div>

                {/* Messages */}
                {selectedPhase && messagesByPhase[selectedPhase] && (
                  <div className="space-y-2 max-h-96 overflow-y-auto">
                    {messagesByPhase[selectedPhase].map((msg, idx) => (
                      <div
                        key={idx}
                        className="p-3 border rounded-lg bg-background"
                      >
                        <div className="flex items-center justify-between mb-2">
                          <Badge variant="outline">{msg.participantId}</Badge>
                          <span className="text-xs text-muted-foreground">
                            {msg.modelName}
                          </span>
                        </div>
                        <pre className="text-xs overflow-x-auto">
                          {JSON.stringify(msg.content, null, 2)}
                        </pre>
                      </div>
                    ))}
                  </div>
                )}
              </>
            ) : (
              <p className="text-muted-foreground text-center py-8">
                No debate transcript available
              </p>
            )}
          </TabsContent>

          {/* Audit Trail Tab */}
          <TabsContent value="audit" className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <p className="text-sm text-muted-foreground">Snapshot ID</p>
                <p className="font-mono text-sm">{decision.snapshotId}</p>
              </div>
              {decision.universeHash && (
                <div>
                  <p className="text-sm text-muted-foreground">Universe Hash</p>
                  <p className="font-mono text-sm">{decision.universeHash}</p>
                </div>
              )}
              {decision.inputsHash && (
                <div>
                  <p className="text-sm text-muted-foreground">Inputs Hash</p>
                  <p className="font-mono text-sm">{decision.inputsHash}</p>
                </div>
              )}
              <div>
                <p className="text-sm text-muted-foreground">Decision ID</p>
                <p className="font-mono text-sm">{decision.decisionId}</p>
              </div>
            </div>

            {/* Token Usage */}
            {transcript && (transcript.totalInputTokens || transcript.totalOutputTokens) && (
              <div className="p-4 bg-muted rounded-lg">
                <h4 className="font-semibold mb-2">Token Usage</h4>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <p className="text-sm text-muted-foreground">Input Tokens</p>
                    <p className="font-medium">
                      {transcript.totalInputTokens?.toLocaleString()}
                    </p>
                  </div>
                  <div>
                    <p className="text-sm text-muted-foreground">Output Tokens</p>
                    <p className="font-medium">
                      {transcript.totalOutputTokens?.toLocaleString()}
                    </p>
                  </div>
                </div>
              </div>
            )}
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
}

// Simple decision list for showing recent decisions
interface DecisionListProps {
  decisions: DecisionRecord[];
  onDecisionClick?: (decisionId: string) => void;
}

export function DecisionList({ decisions, onDecisionClick }: DecisionListProps) {
  return (
    <div className="space-y-2">
      {decisions.map((decision) => (
        <div
          key={decision.decisionId}
          className="flex items-center justify-between p-3 border rounded-lg hover:bg-muted cursor-pointer"
          onClick={() => onDecisionClick?.(decision.decisionId)}
        >
          <div className="flex items-center gap-3">
            <Badge
              variant={decision.decisionType === "trade" ? "default" : "secondary"}
            >
              {decision.decisionType}
            </Badge>
            <span className="text-sm">
              {formatDate(decision.asofTimestamp)}
            </span>
          </div>
          <Badge className={`${statusColors[decision.status]} text-white`}>
            {decision.status}
          </Badge>
        </div>
      ))}
    </div>
  );
}
