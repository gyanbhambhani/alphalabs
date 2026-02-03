'use client';

import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { TrendingUp } from 'lucide-react';
import type { SharpeEvolutionChartData } from '@/types';
import { cn } from '@/lib/utils';

interface SharpeEvolutionChartProps {
  data: SharpeEvolutionChartData;
  config?: { title?: string; symbol?: string; window?: number };
}

export function SharpeEvolutionChart({ data, config }: SharpeEvolutionChartProps) {
  const { dates, sharpe, current_sharpe, avg_sharpe, thresholds } = data;
  
  // Prepare chart data
  const chartData = dates.map((date, i) => ({
    date,
    sharpe: sharpe[i],
  }));
  
  // Determine quality
  const quality = current_sharpe > 2 
    ? 'Excellent' 
    : current_sharpe > 1 
      ? 'Good' 
      : current_sharpe > 0 
        ? 'Average' 
        : 'Poor';
  
  const qualityColor = current_sharpe > 2 
    ? 'bg-green-500' 
    : current_sharpe > 1 
      ? 'bg-blue-500' 
      : current_sharpe > 0 
        ? 'bg-yellow-500' 
        : 'bg-red-500';
  
  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <TrendingUp className="h-5 w-5" />
            {config?.title || `Rolling Sharpe Ratio`}
            {config?.window && (
              <span className="text-sm text-muted-foreground">
                ({config.window}d)
              </span>
            )}
          </div>
          <Badge className={cn(qualityColor, 'text-white')}>
            {quality}
          </Badge>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Current Stats */}
        <div className="grid grid-cols-2 gap-4">
          <div className="text-center p-3 bg-muted rounded-lg">
            <p className="text-sm text-muted-foreground">Current Sharpe</p>
            <p className={cn(
              'text-2xl font-bold',
              current_sharpe > avg_sharpe ? 'text-green-500' : 'text-orange-500'
            )}>
              {current_sharpe.toFixed(2)}
            </p>
            <p className="text-xs text-muted-foreground">
              {current_sharpe > avg_sharpe ? 'Above' : 'Below'} average
            </p>
          </div>
          <div className="text-center p-3 bg-muted rounded-lg">
            <p className="text-sm text-muted-foreground">Historical Avg</p>
            <p className="text-2xl font-bold">
              {avg_sharpe.toFixed(2)}
            </p>
          </div>
        </div>
        
        {/* Chart */}
        <div className="h-48">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" opacity={0.2} />
              <XAxis 
                dataKey="date" 
                tick={{ fontSize: 10 }} 
                interval="preserveStartEnd"
              />
              <YAxis 
                tick={{ fontSize: 10 }}
                domain={['auto', 'auto']}
              />
              <Tooltip 
                formatter={(value) => [
                  typeof value === 'number' ? value.toFixed(2) : String(value), 
                  'Sharpe'
                ]}
              />
              
              {/* Reference lines for thresholds */}
              {thresholds.includes(0) && (
                <ReferenceLine 
                  y={0} 
                  stroke="#ef4444" 
                  strokeDasharray="3 3"
                />
              )}
              {thresholds.includes(1) && (
                <ReferenceLine 
                  y={1} 
                  stroke="#22c55e" 
                  strokeDasharray="3 3"
                  label={{ value: 'Good', fontSize: 10, fill: '#22c55e' }}
                />
              )}
              {thresholds.includes(2) && (
                <ReferenceLine 
                  y={2} 
                  stroke="#3b82f6" 
                  strokeDasharray="3 3"
                  label={{ value: 'Excellent', fontSize: 10, fill: '#3b82f6' }}
                />
              )}
              
              <Line
                type="monotone"
                dataKey="sharpe"
                stroke="#8884d8"
                strokeWidth={2}
                dot={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
        
        {/* Legend */}
        <div className="flex justify-center gap-4 text-xs text-muted-foreground">
          <div className="flex items-center gap-1">
            <div className="w-3 h-0.5 bg-red-500" />
            <span>Poor (&lt;0)</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-3 h-0.5 bg-green-500" />
            <span>Good (&gt;1)</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-3 h-0.5 bg-blue-500" />
            <span>Excellent (&gt;2)</span>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
