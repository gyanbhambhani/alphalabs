'use client';

import { useEffect, useState } from 'react';
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import type { Embedding, Stock } from '@/types';

interface MultiStockCompareProps {
  symbols: string[];
}

const COLORS = [
  '#22c55e', // green
  '#3b82f6', // blue
  '#f59e0b', // amber
  '#ef4444', // red
  '#8b5cf6', // purple
  '#ec4899', // pink
];

export function MultiStockCompare({ symbols }: MultiStockCompareProps) {
  const [stocksData, setStocksData] = useState<
    Record<string, { stock: Stock; embeddings: Embedding[] }>
  >({});
  const [isLoading, setIsLoading] = useState(true);
  const [metric, setMetric] = useState<'return1m' | 'volatility21d'>('return1m');

  useEffect(() => {
    if (symbols.length > 0) {
      loadStocksData();
    }
  }, [symbols]);

  const loadStocksData = async () => {
    setIsLoading(true);
    try {
      const { api } = await import('@/lib/api');
      
      const data: Record<string, { stock: Stock; embeddings: Embedding[] }> = {};
      
      // Load data for each symbol in parallel
      await Promise.all(
        symbols.map(async (symbol) => {
          try {
            const [stock, embeddingsResponse] = await Promise.all([
              api.getStock(symbol),
              api.getEmbeddingsForSymbol(symbol, {
                perPage: 500,
                sortBy: 'date',
                order: 'asc',
              }),
            ]);
            
            data[symbol] = {
              stock,
              embeddings: embeddingsResponse.embeddings,
            };
          } catch (error) {
            console.error(`Failed to load data for ${symbol}:`, error);
          }
        })
      );
      
      setStocksData(data);
    } catch (error) {
      console.error('Failed to load stocks data:', error);
    } finally {
      setIsLoading(false);
    }
  };

  if (isLoading) {
    return (
      <Card>
        <CardContent className="py-12">
          <div className="text-center text-muted-foreground">
            Loading comparison data...
          </div>
        </CardContent>
      </Card>
    );
  }

  if (symbols.length === 0) {
    return (
      <Card>
        <CardContent className="py-12">
          <div className="text-center text-muted-foreground">
            Select multiple stocks to compare
          </div>
        </CardContent>
      </Card>
    );
  }

  // Prepare chart data
  const prepareChartData = () => {
    const allDates = new Set<string>();
    
    // Collect all unique dates
    Object.values(stocksData).forEach(({ embeddings }) => {
      embeddings.forEach((emb) => allDates.add(emb.metadata.date));
    });
    
    const sortedDates = Array.from(allDates).sort();
    
    // Build chart data
    return sortedDates.map((date) => {
      const dataPoint: any = { date };
      
      Object.entries(stocksData).forEach(([symbol, { embeddings }]) => {
        const emb = embeddings.find((e) => e.metadata.date === date);
        if (emb) {
          if (metric === 'return1m') {
            dataPoint[symbol] = emb.metadata.return1m * 100;
          } else {
            dataPoint[symbol] = emb.metadata.volatility21d * 100;
          }
        }
      });
      
      return dataPoint;
    });
  };

  const chartData = prepareChartData();

  // Calculate comparison stats
  const calculateStats = () => {
    return symbols.map((symbol, idx) => {
      const data = stocksData[symbol];
      if (!data) return null;
      
      const { stock, embeddings } = data;
      
      const returns = embeddings.map((e) => e.metadata.return1m);
      const vols = embeddings.map((e) => e.metadata.volatility21d);
      
      const avgReturn = returns.reduce((a, b) => a + b, 0) / returns.length;
      const avgVol = vols.reduce((a, b) => a + b, 0) / vols.length;
      
      return {
        symbol,
        name: stock.name,
        avgReturn: avgReturn * 100,
        avgVol: avgVol * 100,
        dataPoints: embeddings.length,
        color: COLORS[idx % COLORS.length],
      };
    }).filter(Boolean);
  };

  const stats = calculateStats();

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-background border rounded-lg p-3 shadow-lg">
          <p className="font-semibold mb-2">{label}</p>
          {payload.map((entry: any, idx: number) => (
            <p key={idx} className="text-sm" style={{ color: entry.color }}>
              {entry.name}: {entry.value?.toFixed(2)}%
            </p>
          ))}
        </div>
      );
    }
    return null;
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>Multi-Stock Comparison</CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Metric Selector */}
        <Tabs value={metric} onValueChange={(v) => setMetric(v as any)}>
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="return1m">1M Returns</TabsTrigger>
            <TabsTrigger value="volatility21d">21D Volatility</TabsTrigger>
          </TabsList>

          <TabsContent value="return1m" className="mt-6">
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" opacity={0.2} />
                  <XAxis
                    dataKey="date"
                    tick={{ fontSize: 12 }}
                    tickFormatter={(value) => {
                      const date = new Date(value);
                      return date.toLocaleDateString('en-US', {
                        month: 'short',
                        year: '2-digit',
                      });
                    }}
                  />
                  <YAxis
                    tick={{ fontSize: 12 }}
                    label={{
                      value: '1M Return (%)',
                      angle: -90,
                      position: 'insideLeft',
                    }}
                  />
                  <Tooltip content={<CustomTooltip />} />
                  <Legend />
                  {symbols.map((symbol, idx) => (
                    <Line
                      key={symbol}
                      type="monotone"
                      dataKey={symbol}
                      stroke={COLORS[idx % COLORS.length]}
                      dot={false}
                      strokeWidth={2}
                    />
                  ))}
                </LineChart>
              </ResponsiveContainer>
            </div>
          </TabsContent>

          <TabsContent value="volatility21d" className="mt-6">
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" opacity={0.2} />
                  <XAxis
                    dataKey="date"
                    tick={{ fontSize: 12 }}
                    tickFormatter={(value) => {
                      const date = new Date(value);
                      return date.toLocaleDateString('en-US', {
                        month: 'short',
                        year: '2-digit',
                      });
                    }}
                  />
                  <YAxis
                    tick={{ fontSize: 12 }}
                    label={{
                      value: '21D Volatility (%)',
                      angle: -90,
                      position: 'insideLeft',
                    }}
                  />
                  <Tooltip content={<CustomTooltip />} />
                  <Legend />
                  {symbols.map((symbol, idx) => (
                    <Line
                      key={symbol}
                      type="monotone"
                      dataKey={symbol}
                      stroke={COLORS[idx % COLORS.length]}
                      dot={false}
                      strokeWidth={2}
                    />
                  ))}
                </LineChart>
              </ResponsiveContainer>
            </div>
          </TabsContent>
        </Tabs>

        {/* Stats Table */}
        <div className="border rounded-lg overflow-hidden">
          <table className="w-full">
            <thead className="bg-muted">
              <tr>
                <th className="text-left p-3 text-sm font-semibold">Symbol</th>
                <th className="text-left p-3 text-sm font-semibold">Name</th>
                <th className="text-right p-3 text-sm font-semibold">Avg Return</th>
                <th className="text-right p-3 text-sm font-semibold">Avg Vol</th>
                <th className="text-right p-3 text-sm font-semibold">Data Points</th>
              </tr>
            </thead>
            <tbody>
              {stats.map((stat) => (
                <tr key={stat!.symbol} className="border-t">
                  <td className="p-3">
                    <div className="flex items-center gap-2">
                      <div
                        className="w-3 h-3 rounded-full"
                        style={{ backgroundColor: stat!.color }}
                      />
                      <span className="font-mono font-semibold">
                        {stat!.symbol}
                      </span>
                    </div>
                  </td>
                  <td className="p-3 text-sm text-muted-foreground">
                    {stat!.name}
                  </td>
                  <td className="p-3 text-right font-mono">
                    <span className={stat!.avgReturn >= 0 ? 'text-green-500' : 'text-red-500'}>
                      {stat!.avgReturn >= 0 ? '+' : ''}
                      {stat!.avgReturn.toFixed(2)}%
                    </span>
                  </td>
                  <td className="p-3 text-right font-mono">
                    {stat!.avgVol.toFixed(2)}%
                  </td>
                  <td className="p-3 text-right font-mono">
                    {stat!.dataPoints.toLocaleString()}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </CardContent>
    </Card>
  );
}
