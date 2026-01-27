'use client';

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import {
  FileText,
  Target,
  AlertTriangle,
  TrendingUp,
  TrendingDown,
  Clock,
  BarChart2,
} from 'lucide-react';
import ReactMarkdown from 'react-markdown';

interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
  timestamp: string;
  queryType?: string;
  data?: any;
  suggestions?: string[];
}

interface MarketContext {
  symbol: string;
  regime: string;
  volatility: number;
  momentum_1m: number;
  recommendation: string;
  confidence: number;
  interpretation: string;
  avg_forward_return_1m?: number;
  positive_outcome_rate?: number;
  worst_case_drawdown?: number;
  key_risks?: string[];
  similar_periods: Array<{
    date: string;
    similarity: number;
    regime: string;
    narrative: string;
  }>;
}

interface ThesisViewerProps {
  messages: ChatMessage[];
  context: MarketContext | null;
}

export function ThesisViewer({ messages, context }: ThesisViewerProps) {
  // Find the most recent research message
  const researchMessages = messages.filter(
    (m) =>
      m.role === 'assistant' &&
      (m.queryType === 'research' ||
        m.queryType === 'stock_analysis' ||
        m.content.includes('## ') ||
        m.content.includes('Research Report'))
  );

  const latestResearch = researchMessages[researchMessages.length - 1];

  if (!latestResearch && !context) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-center py-12">
        <FileText className="h-12 w-12 text-muted-foreground mb-4" />
        <h3 className="text-lg font-medium mb-2">No Analysis Yet</h3>
        <p className="text-sm text-muted-foreground max-w-xs">
          Ask for a research report or stock analysis to see detailed insights
          here.
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Research Content */}
      {latestResearch && (
        <Card>
          <CardHeader className="pb-2">
            <div className="flex items-center justify-between">
              <CardTitle className="text-sm font-medium flex items-center gap-2">
                <FileText className="h-4 w-4" />
                Latest Analysis
              </CardTitle>
              {latestResearch.queryType && (
                <Badge variant="outline" className="text-xs">
                  {latestResearch.queryType.replace('_', ' ')}
                </Badge>
              )}
            </div>
          </CardHeader>
          <CardContent>
            <div className="prose prose-sm dark:prose-invert max-w-none">
              <ReactMarkdown
                components={{
                  table: ({ children }) => (
                    <div className="overflow-x-auto my-2">
                      <table className="min-w-full text-xs border-collapse">
                        {children}
                      </table>
                    </div>
                  ),
                  th: ({ children }) => (
                    <th className="border border-border px-2 py-1 bg-muted font-medium text-left">
                      {children}
                    </th>
                  ),
                  td: ({ children }) => (
                    <td className="border border-border px-2 py-1">{children}</td>
                  ),
                  h1: ({ children }) => (
                    <h1 className="text-lg font-bold mt-4 mb-2">{children}</h1>
                  ),
                  h2: ({ children }) => (
                    <h2 className="text-base font-semibold mt-4 mb-2 flex items-center gap-2">
                      <Target className="h-4 w-4" />
                      {children}
                    </h2>
                  ),
                  h3: ({ children }) => (
                    <h3 className="text-sm font-semibold mt-3 mb-1">{children}</h3>
                  ),
                  h4: ({ children }) => (
                    <h4 className="text-sm font-medium mt-2 mb-1">{children}</h4>
                  ),
                  ul: ({ children }) => (
                    <ul className="list-disc pl-4 space-y-1 text-sm">
                      {children}
                    </ul>
                  ),
                  li: ({ children }) => <li className="text-sm">{children}</li>,
                  p: ({ children }) => <p className="text-sm mb-2">{children}</p>,
                  strong: ({ children }) => (
                    <strong className="font-semibold text-foreground">
                      {children}
                    </strong>
                  ),
                  hr: () => <Separator className="my-4" />,
                }}
              >
                {latestResearch.content}
              </ReactMarkdown>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Stats Summary from Data */}
      {latestResearch?.data && (
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium flex items-center gap-2">
              <BarChart2 className="h-4 w-4" />
              Key Metrics
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 gap-4">
              {latestResearch.data.symbol && (
                <div>
                  <p className="text-xs text-muted-foreground">Symbol</p>
                  <p className="font-bold">{latestResearch.data.symbol}</p>
                </div>
              )}
              {latestResearch.data.recommendation && (
                <div>
                  <p className="text-xs text-muted-foreground">Recommendation</p>
                  <Badge
                    className={
                      latestResearch.data.recommendation.includes('long')
                        ? 'bg-green-500/20 text-green-500'
                        : latestResearch.data.recommendation.includes('defensive')
                        ? 'bg-red-500/20 text-red-500'
                        : 'bg-yellow-500/20 text-yellow-500'
                    }
                  >
                    {latestResearch.data.recommendation.replace('_', ' ')}
                  </Badge>
                </div>
              )}
              {latestResearch.data.confidence !== undefined && (
                <div>
                  <p className="text-xs text-muted-foreground">Confidence</p>
                  <p className="font-bold">
                    {(latestResearch.data.confidence * 100).toFixed(0)}%
                  </p>
                </div>
              )}
              {latestResearch.data.avg_forward_return !== undefined && (
                <div>
                  <p className="text-xs text-muted-foreground">Avg Forward Return</p>
                  <p
                    className={`font-bold flex items-center gap-1 ${
                      latestResearch.data.avg_forward_return > 0
                        ? 'text-green-500'
                        : 'text-red-500'
                    }`}
                  >
                    {latestResearch.data.avg_forward_return > 0 ? (
                      <TrendingUp className="h-4 w-4" />
                    ) : (
                      <TrendingDown className="h-4 w-4" />
                    )}
                    {(latestResearch.data.avg_forward_return * 100).toFixed(1)}%
                  </p>
                </div>
              )}
              {latestResearch.data.positive_rate !== undefined && (
                <div>
                  <p className="text-xs text-muted-foreground">Win Rate</p>
                  <p className="font-bold">
                    {(latestResearch.data.positive_rate * 100).toFixed(0)}%
                  </p>
                </div>
              )}
              {latestResearch.data.regime && (
                <div>
                  <p className="text-xs text-muted-foreground">Regime</p>
                  <Badge variant="outline">
                    {latestResearch.data.regime.replace('_', ' ')}
                  </Badge>
                </div>
              )}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Context Summary */}
      {context && !latestResearch && (
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium flex items-center gap-2">
              <Target className="h-4 w-4" />
              {context.symbol} Summary
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <p className="text-xs text-muted-foreground">Regime</p>
                <Badge variant="outline">
                  {context.regime.replace('_', ' ')}
                </Badge>
              </div>
              <div>
                <p className="text-xs text-muted-foreground">Recommendation</p>
                <p className="font-medium">
                  {context.recommendation.replace('_', ' ')}
                </p>
              </div>
              <div>
                <p className="text-xs text-muted-foreground">Confidence</p>
                <p className="font-bold">
                  {(context.confidence * 100).toFixed(0)}%
                </p>
              </div>
              <div>
                <p className="text-xs text-muted-foreground">
                  Historical Win Rate
                </p>
                <p className="font-bold">
                  {((context.positive_outcome_rate || 0) * 100).toFixed(0)}%
                </p>
              </div>
            </div>
            
            <Separator />
            
            <div>
              <p className="text-xs text-muted-foreground mb-2">Interpretation</p>
              <p className="text-sm">{context.interpretation}</p>
            </div>
            
            {context.key_risks && context.key_risks.length > 0 && (
              <>
                <Separator />
                <div>
                  <p className="text-xs text-muted-foreground mb-2 flex items-center gap-1">
                    <AlertTriangle className="h-3 w-3" />
                    Risk Factors
                  </p>
                  <ul className="text-sm space-y-1">
                    {context.key_risks.map((risk, idx) => (
                      <li key={idx} className="text-muted-foreground">
                        • {risk}
                      </li>
                    ))}
                  </ul>
                </div>
              </>
            )}
          </CardContent>
        </Card>
      )}

      {/* Timestamp */}
      {latestResearch && (
        <div className="flex items-center justify-center gap-2 text-xs text-muted-foreground">
          <Clock className="h-3 w-3" />
          {new Date(latestResearch.timestamp).toLocaleString()}
        </div>
      )}
    </div>
  );
}
