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
import { BarChart2 } from 'lucide-react';
import type { ReturnsDistributionChartData } from '@/types';
import { cn } from '@/lib/utils';

interface ReturnsDistributionChartProps {
  data: ReturnsDistributionChartData;
  config?: { title?: string; symbol?: string };
}

export function ReturnsDistributionChart({ 
  data, 
  config 
}: ReturnsDistributionChartProps) {
  const { histogram, stats, recent_returns } = data;
  
  // Prepare histogram data
  const histogramData = histogram.counts.map((count, i) => ({
    range: histogram.labels[i],
    count,
    bin: histogram.bins[i],
    isPositive: (histogram.bins[i]?.[0] ?? 0) >= 0,
  }));
  
  const skew = stats.daily.skew;
  const skewDirection = skew > 0.5 
    ? 'Positive Skew' 
    : skew < -0.5 
      ? 'Negative Skew' 
      : 'Symmetric';
  
  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <BarChart2 className="h-5 w-5" />
            {config?.title || 'Returns Distribution'}
          </div>
          <Badge variant="secondary">
            {skewDirection}
          </Badge>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Recent Returns */}
        <div className="grid grid-cols-3 gap-4">
          <div className="text-center p-3 bg-muted rounded-lg">
            <p className="text-sm text-muted-foreground">1 Day</p>
            <p className={cn(
              'text-xl font-bold',
              recent_returns['1d'] > 0 ? 'text-green-500' : 'text-red-500'
            )}>
              {recent_returns['1d'] > 0 ? '+' : ''}
              {(recent_returns['1d'] * 100).toFixed(2)}%
            </p>
          </div>
          <div className="text-center p-3 bg-muted rounded-lg">
            <p className="text-sm text-muted-foreground">1 Week</p>
            <p className={cn(
              'text-xl font-bold',
              recent_returns['1w'] > 0 ? 'text-green-500' : 'text-red-500'
            )}>
              {recent_returns['1w'] > 0 ? '+' : ''}
              {(recent_returns['1w'] * 100).toFixed(2)}%
            </p>
          </div>
          <div className="text-center p-3 bg-muted rounded-lg">
            <p className="text-sm text-muted-foreground">1 Month</p>
            <p className={cn(
              'text-xl font-bold',
              recent_returns['1m'] > 0 ? 'text-green-500' : 'text-red-500'
            )}>
              {recent_returns['1m'] > 0 ? '+' : ''}
              {(recent_returns['1m'] * 100).toFixed(2)}%
            </p>
          </div>
        </div>
        
        {/* Histogram */}
        <div className="h-48">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={histogramData}>
              <CartesianGrid strokeDasharray="3 3" opacity={0.2} />
              <XAxis 
                dataKey="range" 
                tick={{ fontSize: 9 }} 
                angle={-45}
                textAnchor="end"
                height={60}
                interval={0}
              />
              <YAxis tick={{ fontSize: 10 }} />
              <Tooltip />
              <ReferenceLine x={0} stroke="#888" />
              <Bar dataKey="count" name="Days">
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
        
        {/* Statistics */}
        <div className="grid grid-cols-2 gap-4 text-sm">
          <div>
            <p className="text-muted-foreground">Monthly Stats</p>
            <div className="space-y-1 mt-1">
              <div className="flex justify-between">
                <span>Avg Return:</span>
                <span className="font-mono">
                  {(stats.monthly.mean * 100).toFixed(2)}%
                </span>
              </div>
              <div className="flex justify-between">
                <span>Std Dev:</span>
                <span className="font-mono">
                  {(stats.monthly.std * 100).toFixed(2)}%
                </span>
              </div>
              <div className="flex justify-between">
                <span>Win Rate:</span>
                <span className="font-mono">
                  {(stats.monthly.positive_rate * 100).toFixed(0)}%
                </span>
              </div>
            </div>
          </div>
          <div>
            <p className="text-muted-foreground">Daily Stats</p>
            <div className="space-y-1 mt-1">
              <div className="flex justify-between">
                <span>Best Day:</span>
                <span className="font-mono text-green-500">
                  +{(stats.daily.max * 100).toFixed(2)}%
                </span>
              </div>
              <div className="flex justify-between">
                <span>Worst Day:</span>
                <span className="font-mono text-red-500">
                  {(stats.daily.min * 100).toFixed(2)}%
                </span>
              </div>
              <div className="flex justify-between">
                <span>Skewness:</span>
                <span className="font-mono">
                  {stats.daily.skew.toFixed(2)}
                </span>
              </div>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
