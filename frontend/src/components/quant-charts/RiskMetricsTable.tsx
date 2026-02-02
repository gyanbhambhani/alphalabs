'use client';

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Shield, AlertTriangle } from 'lucide-react';
import type { RiskMetricsTableData } from '@/types';
import { cn } from '@/lib/utils';

interface RiskMetricsTableProps {
  data: RiskMetricsTableData;
}

export function RiskMetricsTable({ data }: RiskMetricsTableProps) {
  const { title, rows } = data;
  
  // Determine risk level from VaR
  const varRow = rows.find(r => r.Metric.includes('VaR'));
  const varValue = varRow ? parseFloat(varRow.Value) : 0;
  const riskLevel = Math.abs(varValue) > 3 
    ? 'high' 
    : Math.abs(varValue) > 2 
      ? 'moderate' 
      : 'low';
  
  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Shield className="h-5 w-5" />
            {title || 'Risk Metrics'}
          </div>
          <Badge 
            variant={riskLevel === 'high' ? 'destructive' : 'secondary'}
            className={cn(
              riskLevel === 'low' && 'bg-green-500 text-white',
              riskLevel === 'moderate' && 'bg-yellow-500 text-black'
            )}
          >
            {riskLevel === 'high' && <AlertTriangle className="h-3 w-3 mr-1" />}
            {riskLevel.charAt(0).toUpperCase() + riskLevel.slice(1)} Risk
          </Badge>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b">
                <th className="text-left py-3 px-4 text-sm font-medium">
                  Metric
                </th>
                <th className="text-right py-3 px-4 text-sm font-medium">
                  Value
                </th>
                <th className="text-left py-3 px-4 text-sm font-medium">
                  Interpretation
                </th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row, idx) => {
                const isNegative = row.Value.includes('-');
                const isDrawdown = row.Metric.includes('Drawdown');
                
                return (
                  <tr 
                    key={idx} 
                    className={cn(
                      'border-b hover:bg-muted/50',
                      idx % 2 === 0 && 'bg-muted/20'
                    )}
                  >
                    <td className="py-3 px-4 text-sm font-medium">
                      {row.Metric}
                    </td>
                    <td className={cn(
                      'py-3 px-4 text-sm font-mono text-right',
                      isDrawdown && isNegative && 'text-red-500',
                      row.Metric.includes('VaR') && 'text-orange-500'
                    )}>
                      {row.Value}
                    </td>
                    <td className="py-3 px-4 text-sm text-muted-foreground">
                      {row.Interpretation}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </CardContent>
    </Card>
  );
}
