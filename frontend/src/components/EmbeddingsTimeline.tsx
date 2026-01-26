'use client';

import { useState } from 'react';
import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from 'recharts';
import { Button } from '@/components/ui/button';
import type { Embedding } from '@/types';

interface EmbeddingsTimelineProps {
  embeddings: Embedding[];
  onPointClick?: (embedding: Embedding) => void;
}

type YAxisMetric = 'return1m' | 'return3m' | 'volatility21d';

export function EmbeddingsTimeline({
  embeddings,
  onPointClick,
}: EmbeddingsTimelineProps) {
  const [yAxis, setYAxis] = useState<YAxisMetric>('return1m');

  if (embeddings.length === 0) {
    return (
      <div className="text-center py-12 text-muted-foreground">
        No data to display
      </div>
    );
  }

  // Prepare data for chart
  const chartData = embeddings.map((embedding) => ({
    date: new Date(embedding.metadata.date).getTime(),
    dateStr: embedding.metadata.date,
    return1m: embedding.metadata.return1m * 100,
    return3m: embedding.metadata.return3m * 100,
    volatility21d: embedding.metadata.volatility21d * 100,
    price: embedding.metadata.price,
    embedding,
  }));

  // Sort by date
  chartData.sort((a, b) => a.date - b.date);

  // Get color based on value
  const getColor = (value: number) => {
    if (yAxis === 'volatility21d') {
      // Volatility: low = green, high = red
      if (value < 15) return '#22c55e'; // green
      if (value > 25) return '#ef4444'; // red
      return '#eab308'; // yellow
    } else {
      // Returns: positive = green, negative = red
      if (value > 2) return '#22c55e'; // green
      if (value < -2) return '#ef4444'; // red
      return '#eab308'; // yellow
    }
  };

  const yAxisLabel = {
    return1m: '1M Return (%)',
    return3m: '3M Return (%)',
    volatility21d: '21D Volatility (%)',
  }[yAxis];

  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      return (
        <div className="bg-background border rounded-lg p-3 shadow-lg">
          <p className="font-semibold">{data.dateStr}</p>
          <p className="text-sm">Price: ${data.price.toFixed(2)}</p>
          <p className="text-sm">
            1M Return: {data.return1m >= 0 ? '+' : ''}
            {data.return1m.toFixed(2)}%
          </p>
          <p className="text-sm">
            3M Return: {data.return3m >= 0 ? '+' : ''}
            {data.return3m.toFixed(2)}%
          </p>
          <p className="text-sm">
            21D Vol: {data.volatility21d.toFixed(2)}%
          </p>
        </div>
      );
    }
    return null;
  };

  return (
    <div className="space-y-4">
      {/* Y-axis selector */}
      <div className="flex items-center gap-2">
        <span className="text-sm text-muted-foreground">Y-Axis:</span>
        <div className="flex gap-1">
          <Button
            size="sm"
            variant={yAxis === 'return1m' ? 'default' : 'outline'}
            onClick={() => setYAxis('return1m')}
          >
            1M Return
          </Button>
          <Button
            size="sm"
            variant={yAxis === 'return3m' ? 'default' : 'outline'}
            onClick={() => setYAxis('return3m')}
          >
            3M Return
          </Button>
          <Button
            size="sm"
            variant={yAxis === 'volatility21d' ? 'default' : 'outline'}
            onClick={() => setYAxis('volatility21d')}
          >
            Volatility
          </Button>
        </div>
      </div>

      {/* Chart */}
      <div className="h-96 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <ScatterChart
            margin={{ top: 20, right: 20, bottom: 60, left: 60 }}
          >
            <CartesianGrid strokeDasharray="3 3" opacity={0.2} />
            <XAxis
              dataKey="date"
              type="number"
              domain={['dataMin', 'dataMax']}
              tickFormatter={(timestamp) => {
                const date = new Date(timestamp);
                return date.toLocaleDateString('en-US', {
                  year: 'numeric',
                  month: 'short',
                });
              }}
              label={{
                value: 'Date',
                position: 'insideBottom',
                offset: -10,
              }}
            />
            <YAxis
              dataKey={yAxis}
              label={{
                value: yAxisLabel,
                angle: -90,
                position: 'insideLeft',
              }}
            />
            <Tooltip content={<CustomTooltip />} />
            <Scatter
              data={chartData}
              onClick={(data) => {
                if (onPointClick && data && data.embedding) {
                  onPointClick(data.embedding);
                }
              }}
              cursor="pointer"
            >
              {chartData.map((entry, index) => (
                <Cell
                  key={`cell-${index}`}
                  fill={getColor(entry[yAxis])}
                />
              ))}
            </Scatter>
          </ScatterChart>
        </ResponsiveContainer>
      </div>

      {/* Legend */}
      <div className="flex items-center justify-center gap-6 text-sm">
        {yAxis === 'volatility21d' ? (
          <>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-green-500" />
              <span>Low (&lt;15%)</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-yellow-500" />
              <span>Normal (15-25%)</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-red-500" />
              <span>High (&gt;25%)</span>
            </div>
          </>
        ) : (
          <>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-green-500" />
              <span>Bullish (&gt;2%)</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-yellow-500" />
              <span>Neutral (-2% to 2%)</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-red-500" />
              <span>Bearish (&lt;-2%)</span>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
