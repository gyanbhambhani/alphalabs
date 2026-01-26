'use client';

import { X, TrendingUp, Activity } from 'lucide-react';
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  CardDescription,
} from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Separator } from '@/components/ui/separator';
import type { Embedding } from '@/types';
import { cn } from '@/lib/utils';

interface EmbeddingDetailProps {
  embedding: Embedding | null;
  onClose: () => void;
  onFindSimilar?: (embedding: Embedding) => void;
}

const MetricRow = ({
  label,
  value,
  icon,
}: {
  label: string;
  value: React.ReactNode;
  icon?: React.ReactNode;
}) => (
  <div className="flex items-center justify-between py-2">
    <div className="flex items-center gap-2 text-sm text-muted-foreground">
      {icon}
      <span>{label}</span>
    </div>
    <div className="font-mono font-semibold">{value}</div>
  </div>
);

export function EmbeddingDetail({
  embedding,
  onClose,
  onFindSimilar,
}: EmbeddingDetailProps) {
  if (!embedding) {
    return null;
  }

  const { metadata } = embedding;

  const formatPercent = (value: number, showSign = true) => {
    const pct = value * 100;
    const color = value >= 0 ? 'text-green-500' : 'text-red-500';
    return (
      <span className={color}>
        {showSign && value >= 0 ? '+' : ''}
        {pct.toFixed(2)}%
      </span>
    );
  };

  const getRegime = () => {
    const vol = metadata.volatility21d;
    const ret = metadata.return1m;

    let volRegime = 'Normal';
    if (vol < 0.15) volRegime = 'Low';
    else if (vol > 0.25) volRegime = 'High';

    let trendRegime = 'Ranging';
    if (ret > 0.02) trendRegime = 'Uptrending';
    else if (ret < -0.02) trendRegime = 'Downtrending';

    return `${volRegime} Volatility, ${trendRegime}`;
  };

  return (
    <Card className="w-full">
      <CardHeader>
        <div className="flex items-start justify-between">
          <div>
            <CardTitle className="text-xl">
              Market State: {metadata.date}
            </CardTitle>
            <CardDescription className="mt-2">
              {getRegime()}
            </CardDescription>
          </div>
          <Button variant="ghost" size="sm" onClick={onClose}>
            <X className="h-4 w-4" />
          </Button>
        </div>
      </CardHeader>

      <CardContent className="space-y-6">
        {/* Price */}
        <div>
          <h3 className="text-sm font-semibold mb-3">Price</h3>
          <div className="text-3xl font-mono font-bold">
            ${metadata.price.toFixed(2)}
          </div>
        </div>

        <Separator />

        {/* Returns */}
        <div>
          <h3 className="text-sm font-semibold mb-3 flex items-center gap-2">
            <TrendingUp className="h-4 w-4" />
            Returns
          </h3>
          <div className="space-y-1">
            <MetricRow
              label="1 Week"
              value={formatPercent(metadata.return1w)}
            />
            <MetricRow
              label="1 Month"
              value={formatPercent(metadata.return1m)}
            />
            <MetricRow
              label="3 Months"
              value={formatPercent(metadata.return3m)}
            />
            <MetricRow
              label="6 Months"
              value={formatPercent(metadata.return6m)}
            />
            <MetricRow
              label="12 Months"
              value={formatPercent(metadata.return12m)}
            />
          </div>
        </div>

        <Separator />

        {/* Volatility */}
        <div>
          <h3 className="text-sm font-semibold mb-3 flex items-center gap-2">
            <Activity className="h-4 w-4" />
            Volatility (Annualized)
          </h3>
          <div className="space-y-1">
            <MetricRow
              label="5 Day"
              value={
                <span>
                  {(metadata.volatility5d * 100).toFixed(2)}%
                </span>
              }
            />
            <MetricRow
              label="10 Day"
              value={
                <span>
                  {(metadata.volatility10d * 100).toFixed(2)}%
                </span>
              }
            />
            <MetricRow
              label="21 Day"
              value={
                <span className={cn(
                  metadata.volatility21d < 0.15 
                    ? 'text-green-500' 
                    : metadata.volatility21d > 0.25 
                    ? 'text-red-500' 
                    : 'text-yellow-500'
                )}>
                  {(metadata.volatility21d * 100).toFixed(2)}%
                </span>
              }
            />
            <MetricRow
              label="63 Day"
              value={
                <span>
                  {(metadata.volatility63d * 100).toFixed(2)}%
                </span>
              }
            />
          </div>
        </div>

        {/* Find Similar Button */}
        {onFindSimilar && (
          <>
            <Separator />
            <Button
              onClick={() => onFindSimilar(embedding)}
              className="w-full"
              variant="outline"
            >
              Find Similar Market Conditions
            </Button>
          </>
        )}
      </CardContent>
    </Card>
  );
}
