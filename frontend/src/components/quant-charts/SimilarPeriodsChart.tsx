'use client';

import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
  ReferenceLine,
} from 'recharts';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { TrendingUp, TrendingDown, History } from 'lucide-react';
import type { SimilarPeriodsChartData } from '@/types';
import { cn } from '@/lib/utils';

interface SimilarPeriodsChartProps {
  data: SimilarPeriodsChartData;
  config?: { title?: string; symbol?: string };
}

export function SimilarPeriodsChart({ data, config }: SimilarPeriodsChartProps) {
  const { periods, positive_rate, positive_count, total_count, avg_forward_1m } = data;
  
  // Prepare histogram data if available
  const histogramData = data.histogram?.counts.map((count, i) => ({
    range: data.histogram?.labels[i] || '',
    count,
    isPositive: (data.histogram?.bins[i]?.[0] ?? 0) >= 0,
  })) || [];
  
  // Determine sentiment
  const isBullish = positive_rate > 0.6;
  const isBearish = positive_rate < 0.4;
  
  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <History className="h-5 w-5" />
            {config?.title || 'Similar Historical Periods'}
          </div>
          <Badge 
            variant={isBullish ? 'default' : isBearish ? 'destructive' : 'secondary'}
            className={cn(
              isBullish && 'bg-green-500',
              isBearish && 'bg-red-500'
            )}
          >
            {isBullish ? 'Bullish Pattern' : isBearish ? 'Bearish Pattern' : 'Neutral'}
          </Badge>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Summary Stats */}
        <div className="grid grid-cols-3 gap-4">
          <div className="text-center p-3 bg-muted rounded-lg">
            <p className="text-sm text-muted-foreground">Similar Periods</p>
            <p className="text-2xl font-bold">{total_count}</p>
          </div>
          <div className="text-center p-3 bg-muted rounded-lg">
            <p className="text-sm text-muted-foreground">Positive Rate</p>
            <div className="flex items-center justify-center gap-1">
              {positive_rate > 0.5 ? (
                <TrendingUp className="h-4 w-4 text-green-500" />
              ) : (
                <TrendingDown className="h-4 w-4 text-red-500" />
              )}
              <p className={cn(
                'text-2xl font-bold',
                positive_rate > 0.5 ? 'text-green-500' : 'text-red-500'
              )}>
                {(positive_rate * 100).toFixed(0)}%
              </p>
            </div>
            <p className="text-xs text-muted-foreground">
              ({positive_count}/{total_count})
            </p>
          </div>
          <div className="text-center p-3 bg-muted rounded-lg">
            <p className="text-sm text-muted-foreground">Avg 1M Return</p>
            <p className={cn(
              'text-2xl font-bold',
              avg_forward_1m > 0 ? 'text-green-500' : 'text-red-500'
            )}>
              {avg_forward_1m > 0 ? '+' : ''}{(avg_forward_1m * 100).toFixed(1)}%
            </p>
          </div>
        </div>
        
        {/* Returns Distribution Histogram */}
        {histogramData.length > 0 && (
          <div className="h-48">
            <p className="text-sm text-muted-foreground mb-2">
              Forward Returns Distribution (1 Month)
            </p>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={histogramData}>
                <CartesianGrid strokeDasharray="3 3" opacity={0.2} />
                <XAxis 
                  dataKey="range" 
                  tick={{ fontSize: 10 }} 
                  angle={-45}
                  textAnchor="end"
                  height={60}
                />
                <YAxis tick={{ fontSize: 10 }} />
                <Tooltip />
                <ReferenceLine x={0} stroke="#888" />
                <Bar dataKey="count" name="Occurrences">
                  {histogramData.map((entry, index) => (
                    <Cell 
                      key={`cell-${index}`}
                      fill={entry.isPositive ? '#22c55e' : '#ef4444'}
                      opacity={0.8}
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}
        
        {/* Historical Periods Table */}
        <div>
          <p className="text-sm text-muted-foreground mb-2">
            Most Similar Periods
          </p>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b">
                  <th className="text-left py-2 px-2">Date</th>
                  <th className="text-right py-2 px-2">Similarity</th>
                  <th className="text-right py-2 px-2">1M Return</th>
                  <th className="text-left py-2 px-2">Outcome</th>
                </tr>
              </thead>
              <tbody>
                {periods.slice(0, 8).map((period, idx) => (
                  <tr 
                    key={period.date} 
                    className={cn(
                      'border-b hover:bg-muted/50',
                      idx < 3 && 'bg-primary/5'
                    )}
                  >
                    <td className="py-2 px-2 font-mono text-xs">
                      {period.date}
                    </td>
                    <td className="text-right py-2 px-2">
                      <Badge variant="outline" className="text-xs">
                        {(period.similarity * 100).toFixed(1)}%
                      </Badge>
                    </td>
                    <td className={cn(
                      'text-right py-2 px-2 font-mono text-xs',
                      period.forward_1m !== null && period.forward_1m > 0 
                        ? 'text-green-500' 
                        : 'text-red-500'
                    )}>
                      {period.forward_1m !== null 
                        ? `${period.forward_1m > 0 ? '+' : ''}${(period.forward_1m * 100).toFixed(1)}%`
                        : 'N/A'
                      }
                    </td>
                    <td className="py-2 px-2 text-xs text-muted-foreground">
                      {period.outcome}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
