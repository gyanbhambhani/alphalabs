"use client";

import { Fund, FundStrategy } from "@/types";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

interface FundCardProps {
  fund: Fund;
  onClick?: () => void;
}

const strategyLabels: Record<FundStrategy, string> = {
  trend_macro: "Trend + Macro",
  mean_reversion: "Mean Reversion",
  event_driven: "Event-Driven",
  quality_ls: "Quality L/S",
};

const strategyColors: Record<FundStrategy, string> = {
  trend_macro: "bg-blue-500",
  mean_reversion: "bg-green-500",
  event_driven: "bg-orange-500",
  quality_ls: "bg-purple-500",
};

function formatCurrency(value: number): string {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    minimumFractionDigits: 0,
    maximumFractionDigits: 0,
  }).format(value);
}

function formatPercent(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

export function FundCard({ fund, onClick }: FundCardProps) {
  const pnl = fund.totalValue - 100000; // Assuming 100k starting capital
  const pnlPercent = pnl / 100000;
  const isProfitable = pnl >= 0;

  return (
    <Card
      className={`cursor-pointer transition-all hover:shadow-lg ${
        !fund.isActive ? "opacity-60" : ""
      }`}
      onClick={onClick}
    >
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="text-lg font-semibold">{fund.name}</CardTitle>
          <Badge
            className={`${strategyColors[fund.strategy]} text-white text-xs`}
          >
            {strategyLabels[fund.strategy]}
          </Badge>
        </div>
        {!fund.isActive && (
          <Badge variant="outline" className="w-fit text-xs">
            Inactive
          </Badge>
        )}
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Total Value - Primary metric */}
        <div className="text-center py-2">
          <p className="text-3xl font-bold">{formatCurrency(fund.totalValue)}</p>
          <p
            className={`text-sm font-medium ${
              isProfitable ? "text-green-600" : "text-red-600"
            }`}
          >
            {isProfitable ? "+" : ""}
            {formatCurrency(pnl)} ({formatPercent(pnlPercent)})
          </p>
        </div>

        {/* Key Metrics */}
        <div className="grid grid-cols-2 gap-4 text-sm">
          <div>
            <p className="text-muted-foreground">Positions</p>
            <p className="font-medium">{fund.nPositions}</p>
          </div>
          <div>
            <p className="text-muted-foreground">Cash</p>
            <p className="font-medium">{formatCurrency(fund.cashBalance)}</p>
          </div>
          <div>
            <p className="text-muted-foreground">Gross Exposure</p>
            <p className="font-medium">{formatPercent(fund.grossExposure)}</p>
          </div>
          <div>
            <p className="text-muted-foreground">Net Exposure</p>
            <p className="font-medium">{formatPercent(fund.netExposure)}</p>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

// Fund list component for the leaderboard
interface FundLeaderboardProps {
  funds: Fund[];
  onFundClick?: (fundId: string) => void;
}

export function FundLeaderboard({ funds, onFundClick }: FundLeaderboardProps) {
  // Sort by total value
  const sortedFunds = [...funds].sort((a, b) => b.totalValue - a.totalValue);

  return (
    <div className="space-y-4">
      <h2 className="text-xl font-bold">Fund Leaderboard</h2>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {sortedFunds.map((fund, index) => (
          <div key={fund.fundId} className="relative">
            <div className="absolute -top-2 -left-2 w-8 h-8 rounded-full bg-primary text-primary-foreground flex items-center justify-center font-bold text-sm z-10">
              #{index + 1}
            </div>
            <FundCard
              fund={fund}
              onClick={() => onFundClick?.(fund.fundId)}
            />
          </div>
        ))}
      </div>
    </div>
  );
}
