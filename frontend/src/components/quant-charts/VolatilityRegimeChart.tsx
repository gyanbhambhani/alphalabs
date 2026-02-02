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
  Area,
  ComposedChart,
} from 'recharts';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Activity } from 'lucide-react';
import type { VolatilityRegimeChartData } from '@/types';
import { cn } from '@/lib/utils';

interface VolatilityRegimeChartProps {
  data: VolatilityRegimeChartData;
  config?: { title?: string; symbol?: string };
}

export function VolatilityRegimeChart({ data, config }: VolatilityRegimeChartProps) {
  const { 
    dates, 
    volatility, 
    regimes, 
    current_regime, 
    current_vol, 
    avg_vol 
  } = data;
  
  // Prepare chart data
  const chartData = dates.map((date, i) => ({
    date,
    volatility: volatility[i] * 100, // Convert to percentage
    regime: regimes[i],
  }));
  
  const regimeColor = current_regime === 'High' 
    ? 'destructive' 
    : current_regime === 'Low' 
      ? 'default' 
      : 'secondary';
  
  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Activity className="h-5 w-5" />
            {config?.title || `${config?.symbol || ''} Volatility Regime`}
          </div>
          <Badge variant={regimeColor}>
            {current_regime} Vol
          </Badge>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Current Stats */}
        <div className="grid grid-cols-2 gap-4">
          <div className="text-center p-3 bg-muted rounded-lg">
            <p className="text-sm text-muted-foreground">Current Volatility</p>
            <p className={cn(
              'text-2xl font-bold',
              current_vol > avg_vol ? 'text-orange-500' : 'text-blue-500'
            )}>
              {(current_vol * 100).toFixed(1)}%
            </p>
            <p className="text-xs text-muted-foreground">annualized</p>
          </div>
          <div className="text-center p-3 bg-muted rounded-lg">
            <p className="text-sm text-muted-foreground">Average Volatility</p>
            <p className="text-2xl font-bold">
              {(avg_vol * 100).toFixed(1)}%
            </p>
            <p className="text-xs text-muted-foreground">historical</p>
          </div>
        </div>
        
        {/* Chart */}
        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" opacity={0.2} />
              <XAxis 
                dataKey="date" 
                tick={{ fontSize: 10 }} 
                interval="preserveStartEnd"
              />
              <YAxis 
                tick={{ fontSize: 10 }}
                tickFormatter={(val) => `${val}%`}
                domain={['auto', 'auto']}
              />
              <Tooltip 
                formatter={(value: number) => [`${value.toFixed(1)}%`, 'Volatility']}
                labelFormatter={(label) => `Date: ${label}`}
              />
              <ReferenceLine 
                y={avg_vol * 100} 
                stroke="#888" 
                strokeDasharray="3 3"
                label={{ value: 'Avg', fontSize: 10 }}
              />
              <Area
                type="monotone"
                dataKey="volatility"
                fill="#8884d8"
                fillOpacity={0.2}
                stroke="none"
              />
              <Line
                type="monotone"
                dataKey="volatility"
                stroke="#8884d8"
                strokeWidth={2}
                dot={false}
              />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
        
        {/* Recent Regime Changes */}
        {data.regime_changes && data.regime_changes.length > 0 && (
          <div>
            <p className="text-sm text-muted-foreground mb-2">
              Recent Regime Changes
            </p>
            <div className="flex flex-wrap gap-2">
              {data.regime_changes.slice(-5).map((date) => (
                <Badge key={date} variant="outline" className="text-xs">
                  {date}
                </Badge>
              ))}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
