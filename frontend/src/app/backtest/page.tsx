'use client';

import { useEffect, useState, useRef, useCallback } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';

// Types
interface FundRanking {
  fund_id: string;
  fund_name: string;
  rank: number;
  total_value: number;
  cumulative_return: number;
  sharpe_ratio: number;
  max_drawdown: number;
}

interface Decision {
  fund_id: string;
  fund_name: string;
  action: string;
  symbol?: string;
  reasoning: string;
  confidence: number;
  simulation_date: string;
}

interface DebateMessage {
  fund_id: string;
  phase: string;
  model: string;
  content: string;
}

interface PortfolioState {
  fund_id: string;
  fund_name: string;
  total_value: number;
  cash: number;
  cumulative_return: number;
  max_drawdown: number;
  sharpe_ratio: number;
  positions: Record<string, { 
    quantity: number; 
    current_price: number; 
    current_value: number 
  }>;
}

interface FundConfig {
  id: string;
  name: string;
  thesis: string;
  icon: string;
  color: string;
  enabled: boolean;
  initialCash: number;
}

interface SimulationState {
  isRunning: boolean;
  isPaused: boolean;
  simulationId: string | null;
  currentDate: string | null;
  progress: number;
  speed: number;
  benchmarkReturn: number;
  leaderboard: FundRanking[];
  decisions: Decision[];
  debates: DebateMessage[];
  portfolios: Record<string, PortfolioState>;
}

// Default fund configurations
const DEFAULT_FUNDS: FundConfig[] = [
  {
    id: 'momentum',
    name: 'Momentum Fund',
    thesis: `Buy stocks with strong 12-month momentum, skip the most recent month. 
Focus on tech and growth stocks. Sell when momentum weakens.`,
    icon: '',
    color: 'blue',
    enabled: true,
    initialCash: 100000,
  },
  {
    id: 'value',
    name: 'Value Fund',
    thesis: `Buy undervalued stocks with low P/E ratios and strong fundamentals. 
Focus on financials and industrials. Patient, long-term holding.`,
    icon: '',
    color: 'emerald',
    enabled: true,
    initialCash: 100000,
  },
  {
    id: 'mean_reversion',
    name: 'Mean Reversion Fund',
    thesis: `Buy oversold stocks (RSI < 30, negative 1-month returns). 
Sell when they revert to mean. Quick trades, high turnover.`,
    icon: '',
    color: 'violet',
    enabled: true,
    initialCash: 100000,
  },
  {
    id: 'quant',
    name: 'Quant Fund',
    thesis: `Combine multiple signals: momentum, mean reversion, volatility. 
Trade based on signal strength, not gut feel. Systematic approach.`,
    icon: '',
    color: 'amber',
    enabled: false,
    initialCash: 100000,
  },
];

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

// Color themes for funds - simple borders only
const COLOR_THEMES: Record<string, { 
  border: string; 
  text: string;
}> = {
  blue: {
    border: 'border-blue-600',
    text: 'text-blue-400',
  },
  emerald: {
    border: 'border-emerald-600',
    text: 'text-emerald-400',
  },
  violet: {
    border: 'border-violet-600',
    text: 'text-violet-400',
  },
  amber: {
    border: 'border-amber-600',
    text: 'text-amber-400',
  },
};

const DEFAULT_THEME = {
  border: 'border-zinc-700',
  text: 'text-zinc-400',
};

// Map fund IDs to their configs for theme lookup
function getFundTheme(fundId: string, funds: FundConfig[]) {
  const fund = funds.find(f => `${f.id}_fund` === fundId || f.id === fundId);
  if (fund) {
    return COLOR_THEMES[fund.color] || DEFAULT_THEME;
  }
  return DEFAULT_THEME;
}

function formatCurrency(value: number): string {
  if (value >= 1000000) {
    return `$${(value / 1000000).toFixed(2)}M`;
  }
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: 0,
    maximumFractionDigits: 0,
  }).format(value);
}

function formatPercent(value: number): string {
  const pct = value * 100;
  const sign = pct >= 0 ? '+' : '';
  return `${sign}${pct.toFixed(2)}%`;
}

function formatDate(dateStr: string): string {
  const date = new Date(dateStr);
  return date.toLocaleDateString('en-US', { 
    year: 'numeric', 
    month: 'short', 
    day: 'numeric' 
  });
}

// Check if reasoning contains a parse error
function isParseError(reasoning: string): boolean {
  return reasoning.toLowerCase().includes('parse error') || 
         reasoning.toLowerCase().includes('extra data:');
}

export default function BacktestPage() {
  const [state, setState] = useState<SimulationState>({
    isRunning: false,
    isPaused: false,
    simulationId: null,
    currentDate: null,
    progress: 0,
    speed: 10,
    benchmarkReturn: 0,
    leaderboard: [],
    decisions: [],
    debates: [],
    portfolios: {},
  });
  
  const [funds, setFunds] = useState<FundConfig[]>(DEFAULT_FUNDS);
  const [showConfig, setShowConfig] = useState(true);
  const [editingFund, setEditingFund] = useState<string | null>(null);
  const [selectedFund, setSelectedFund] = useState<string | null>(null);
  const [dataReady, setDataReady] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [downloadProgress, setDownloadProgress] = useState<{
    status: string;
    current: number;
    total: number;
    symbol: string;
  } | null>(null);
  
  // Historical runs
  interface HistoricalRun {
    id: string;
    status: string;
    total_trades: number;
    total_decisions: number;
    created_at: string;
    fund_ids: string[];
  }
  const [historicalRuns, setHistoricalRuns] = useState<HistoricalRun[]>([]);
  const [showHistory, setShowHistory] = useState(false);
  
  const eventSourceRef = useRef<EventSource | null>(null);
  const decisionsEndRef = useRef<HTMLDivElement>(null);

  // Check if data is ready and fetch historical runs
  useEffect(() => {
    checkDataStatus();
    fetchHistoricalRuns();
  }, []);
  
  const fetchHistoricalRuns = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/backtest/history/runs`);
      const data = await res.json();
      setHistoricalRuns(data.runs || []);
    } catch (error) {
      console.error('Failed to fetch historical runs:', error);
    }
  };

  const checkDataStatus = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/backtest/data/status`);
      const data = await res.json();
      setDataReady(data.ready);
    } catch (error) {
      console.error('Failed to check data status:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const downloadData = async () => {
    setIsLoading(true);
    try {
      await fetch(`${API_BASE}/api/backtest/data/download`, { method: 'POST' });
      
      // Poll for progress
      const pollProgress = setInterval(async () => {
        try {
          const progressRes = await fetch(`${API_BASE}/api/backtest/data/progress`);
          const progress = await progressRes.json();
          setDownloadProgress(progress);
          
          if (progress.status === 'complete') {
            clearInterval(pollProgress);
            setDataReady(true);
            setIsLoading(false);
            setDownloadProgress(null);
          }
        } catch {
          // Ignore polling errors
        }
      }, 1000);
      
    } catch (error) {
      console.error('Failed to download data:', error);
      setIsLoading(false);
    }
  };

  const startSimulation = async () => {
    try {
      // Build query params from enabled funds
      const enabledFunds = funds.filter(f => f.enabled);
      const templateParams = enabledFunds
        .map(f => `template_ids=${f.id}`)
        .join('&');
      
      const res = await fetch(
        `${API_BASE}/api/backtest/quick-start?` + 
        `${templateParams}&speed=${state.speed}&initial_cash=100000`,
        { method: 'POST' }
      );
      const data = await res.json();
      
      setState(prev => ({
        ...prev,
        simulationId: data.simulation_id,
        isRunning: true,
        decisions: [],
        debates: [],
        leaderboard: [],
      }));
      
      // Hide config panel when simulation starts
      setShowConfig(false);
      
      // Connect to SSE stream
      connectToStream(data.simulation_id);
    } catch (error) {
      console.error('Failed to start simulation:', error);
    }
  };
  
  // Fund configuration helpers
  const updateFund = (id: string, updates: Partial<FundConfig>) => {
    setFunds(prev => prev.map(f => f.id === id ? { ...f, ...updates } : f));
  };
  
  const toggleFund = (id: string) => {
    setFunds(prev => prev.map(f => f.id === id ? { ...f, enabled: !f.enabled } : f));
  };

  const connectToStream = (simulationId: string) => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
    }
    
    const eventSource = new EventSource(
      `${API_BASE}/api/backtest/stream/${simulationId}`
    );
    
    eventSource.onmessage = (event) => {
      const data = JSON.parse(event.data);
      handleEvent(data);
    };
    
    eventSource.onerror = () => {
      console.error('SSE connection error');
      setState(prev => ({ ...prev, isRunning: false }));
    };
    
    eventSourceRef.current = eventSource;
  };

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const handleEvent = useCallback((event: Record<string, any>) => {
    const eventType = event.event_type;
    
    switch (eventType) {
      case 'day_tick':
        setState(prev => ({
          ...prev,
          currentDate: event.simulation_date,
          progress: event.progress_pct,
          benchmarkReturn: event.benchmark_return || 0,
        }));
        break;
        
      case 'decision':
        setState(prev => ({
          ...prev,
          decisions: [
            {
              fund_id: event.fund_id,
              fund_name: event.fund_name,
              action: event.action,
              symbol: event.symbol,
              reasoning: event.reasoning,
              confidence: event.confidence,
              simulation_date: event.simulation_date,
            },
            ...prev.decisions.slice(0, 99), // Keep last 100
          ],
        }));
        break;
        
      case 'debate_message':
        setState(prev => ({
          ...prev,
          debates: [
            {
              fund_id: event.fund_id,
              phase: event.phase,
              model: event.model,
              content: event.content,
            },
            ...prev.debates.slice(0, 49), // Keep last 50
          ],
        }));
        break;
        
      case 'portfolio_update':
        setState(prev => ({
          ...prev,
          portfolios: {
            ...prev.portfolios,
            [event.fund_id]: {
              fund_id: event.fund_id,
              fund_name: event.fund_name,
              total_value: event.total_value,
              cash: event.cash,
              cumulative_return: event.cumulative_return,
              max_drawdown: event.max_drawdown,
              sharpe_ratio: event.sharpe_ratio,
              positions: event.positions || {},
            },
          },
        }));
        break;
        
      case 'leaderboard':
        setState(prev => ({
          ...prev,
          leaderboard: event.rankings || [],
        }));
        break;
        
      case 'simulation_end':
        setState(prev => ({
          ...prev,
          isRunning: false,
          leaderboard: event.final_rankings || prev.leaderboard,
        }));
        if (eventSourceRef.current) {
          eventSourceRef.current.close();
        }
        break;
    }
  }, []);

  const controlSimulation = async (action: string, speed?: number) => {
    if (!state.simulationId) return;
    
    try {
      await fetch(
        `${API_BASE}/api/backtest/control/${state.simulationId}`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ action, speed }),
        }
      );
      
      if (action === 'pause') {
        setState(prev => ({ ...prev, isPaused: true }));
      } else if (action === 'resume') {
        setState(prev => ({ ...prev, isPaused: false }));
      } else if (action === 'stop') {
        setState(prev => ({ ...prev, isRunning: false, isPaused: false }));
        if (eventSourceRef.current) {
          eventSourceRef.current.close();
        }
      } else if (action === 'speed' && speed) {
        setState(prev => ({ ...prev, speed }));
      }
    } catch (error) {
      console.error('Failed to control simulation:', error);
    }
  };

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
      }
    };
  }, []);

  // Scroll decisions to bottom
  useEffect(() => {
    decisionsEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [state.decisions]);
  
  // Refresh historical runs when simulation ends
  useEffect(() => {
    if (!state.isRunning && state.simulationId) {
      // Delay slightly to ensure backend has finished writing
      const timer = setTimeout(() => {
        fetchHistoricalRuns();
      }, 1000);
      return () => clearTimeout(timer);
    }
  }, [state.isRunning, state.simulationId]);

  const selectedPortfolio = selectedFund ? state.portfolios[selectedFund] : null;

  return (
    <div className="min-h-screen bg-black">
      <div className="max-w-[1600px] mx-auto p-6 space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-white">
              AI Fund Time Machine
            </h1>
            <p className="text-zinc-500 mt-1">
              Watch AI-powered funds compete across 25 years of market history
            </p>
          </div>
          {state.isRunning && (
            <div className="flex items-center gap-3">
              <div className="h-2 w-2 rounded-full bg-green-500 animate-pulse" />
              <span className="text-sm text-zinc-400">Live Simulation</span>
            </div>
          )}
        </div>

        {/* Data Status / Controls */}
        {!dataReady ? (
          <Card className="bg-zinc-900 border-zinc-800">
            <CardContent className="py-12 text-center">
              <h3 className="text-lg font-semibold text-white mb-2">
                One-Time Data Download
              </h3>
              <p className="mb-6 text-zinc-400 max-w-md mx-auto">
                Download 25 years of historical stock data (2000-2025). 
                This only needs to be done once - data is cached locally.
              </p>
              
              {downloadProgress && downloadProgress.status === 'downloading' ? (
                <div className="space-y-4">
                  <div className="flex items-center justify-center gap-3">
                    <span className="text-sm text-zinc-300">
                      Downloading {downloadProgress.symbol}
                    </span>
                    <span className="font-mono text-zinc-500">
                      {downloadProgress.current}/{downloadProgress.total}
                    </span>
                  </div>
                  <div className="w-80 mx-auto h-1.5 bg-zinc-800 rounded-full overflow-hidden">
                    <div 
                      className="h-full bg-white transition-all duration-300"
                      style={{ 
                        width: `${(downloadProgress.current / downloadProgress.total) * 100}%` 
                      }}
                    />
                  </div>
                </div>
              ) : (
                <Button 
                  onClick={downloadData} 
                  disabled={isLoading} 
                  size="lg"
                  className="bg-white text-black hover:bg-zinc-200"
                >
                  {isLoading ? 'Starting...' : 'Download Historical Data'}
                </Button>
              )}
              
              <p className="mt-6 text-xs text-zinc-600">
                ~34 stocks x 25 years of daily OHLCV data
              </p>
            </CardContent>
          </Card>
        ) : (
        <>
          {/* Fund Configuration Panel */}
          {showConfig && !state.isRunning && (
            <Card className="bg-zinc-900 border-zinc-800">
              <CardHeader className="pb-4 border-b border-zinc-800">
                <div className="flex items-center justify-between">
                  <CardTitle className="text-lg font-medium text-white">
                    Configure Your Funds
                  </CardTitle>
                  <p className="text-sm text-zinc-500">
                    Edit theses, enable/disable funds, customize strategies
                  </p>
                </div>
              </CardHeader>
              <CardContent className="p-4">
                <div className="grid grid-cols-2 gap-4">
                  {funds.map((fund) => {
                    const theme = COLOR_THEMES[fund.color] || DEFAULT_THEME;
                    const isEditing = editingFund === fund.id;
                    
                    return (
                      <div
                        key={fund.id}
                        className={cn(
                          "rounded-lg border",
                          fund.enabled 
                            ? `bg-zinc-900 ${theme.border}` 
                            : "bg-zinc-900/50 border-zinc-800 opacity-60"
                        )}
                      >
                        {/* Fund Header */}
                        <div className="p-4 border-b border-zinc-800">
                          <div className="flex items-center justify-between">
                            <div>
                              <h3 className="font-semibold text-white">
                                {fund.name}
                              </h3>
                              <p className="text-xs text-zinc-500">
                                ${fund.initialCash.toLocaleString()} initial
                              </p>
                            </div>
                            <div className="flex items-center gap-2">
                              <Button
                                variant="ghost"
                                size="sm"
                                onClick={() => setEditingFund(
                                  isEditing ? null : fund.id
                                )}
                                className="text-zinc-400 hover:text-white h-8 px-2"
                              >
                                {isEditing ? 'Done' : 'Edit'}
                              </Button>
                              <button
                                onClick={() => toggleFund(fund.id)}
                                className={cn(
                                  "w-12 h-6 rounded-full transition-all relative",
                                  fund.enabled 
                                    ? "bg-green-600" 
                                    : "bg-zinc-700"
                                )}
                              >
                                <div className={cn(
                                  "absolute top-1 w-4 h-4 rounded-full bg-white",
                                  "transition-all shadow-sm",
                                  fund.enabled ? "left-7" : "left-1"
                                )} />
                              </button>
                            </div>
                          </div>
                        </div>
                        
                        {/* Fund Thesis */}
                        <div className="p-4">
                          <label className="text-xs text-zinc-500 uppercase 
                                            tracking-wide mb-2 block">
                            Investment Thesis
                          </label>
                          {isEditing ? (
                            <textarea
                              value={fund.thesis}
                              onChange={(e) => updateFund(fund.id, { 
                                thesis: e.target.value 
                              })}
                              className="w-full h-32 bg-zinc-800 border 
                                         border-zinc-700 rounded-lg p-3 text-sm 
                                         text-zinc-300 resize-none focus:outline-none 
                                         focus:border-zinc-600"
                              placeholder="Describe the fund's investment strategy..."
                            />
                          ) : (
                            <p className="text-sm text-zinc-400 leading-relaxed 
                                          whitespace-pre-line">
                              {fund.thesis}
                            </p>
                          )}
                        </div>
                      </div>
                    );
                  })}
                </div>
                
                {/* Summary */}
                <div className="mt-4 pt-4 border-t border-zinc-800 flex items-center 
                                justify-between">
                  <div className="text-sm text-zinc-500">
                    <span className="text-white font-medium">
                      {funds.filter(f => f.enabled).length}
                    </span>
                    {' '}funds enabled, 
                    <span className="text-white font-medium ml-1">
                      ${funds
                        .filter(f => f.enabled)
                        .reduce((sum, f) => sum + f.initialCash, 0)
                        .toLocaleString()}
                    </span>
                    {' '}total capital
                  </div>
                  <Button
                    onClick={() => setShowConfig(false)}
                    variant="ghost"
                    className="text-zinc-400 hover:text-white"
                  >
                    Collapse
                  </Button>
                </div>
              </CardContent>
            </Card>
          )}
          
          {/* Collapsed Config Toggle */}
          {!showConfig && !state.isRunning && (
            <Button
              onClick={() => setShowConfig(true)}
              variant="outline"
              className="w-full border-zinc-700 bg-zinc-900 hover:bg-zinc-800 
                         text-zinc-400 hover:text-white"
            >
              Configure Funds ({funds.filter(f => f.enabled).length} enabled)
            </Button>
          )}
          {/* Time Controls */}
          <Card className="bg-zinc-900 border-zinc-800">
            <CardContent className="py-5">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-4">
                  {!state.isRunning ? (
                    <Button 
                      onClick={startSimulation} 
                      size="lg"
                      className="bg-white text-black hover:bg-zinc-200"
                    >
                      Start Simulation
                    </Button>
                  ) : (
                    <div className="flex items-center gap-2">
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => 
                          controlSimulation(state.isPaused ? 'resume' : 'pause')
                        }
                        className="border-zinc-700 hover:bg-zinc-800"
                      >
                        {state.isPaused ? 'Resume' : 'Pause'}
                      </Button>
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => controlSimulation('stop')}
                        className="border-zinc-700 text-red-400 hover:bg-zinc-800"
                      >
                        Stop
                      </Button>
                    </div>
                  )}
                  
                  <div className="flex items-center gap-1.5 ml-4">
                    <span className="text-xs text-zinc-500 mr-2">Speed</span>
                    {[
                      { value: 1, label: '1x' },
                      { value: 10, label: '10x' },
                      { value: 50, label: '50x' },
                      { value: 100, label: '100x' },
                      { value: 500, label: '500x' },
                    ].map(({ value, label }) => (
                      <Button
                        key={value}
                        variant="ghost"
                        size="sm"
                        onClick={() => {
                          setState(prev => ({ ...prev, speed: value }));
                          if (state.isRunning) {
                            controlSimulation('speed', value);
                          }
                        }}
                        className={cn(
                          "h-8 px-3 text-xs font-mono",
                          state.speed === value 
                            ? "bg-zinc-700 text-white" 
                            : "text-zinc-500 hover:text-white hover:bg-zinc-800"
                        )}
                      >
                        {label}
                      </Button>
                    ))}
                  </div>
                </div>
                
                <div className="flex items-center gap-8">
                  <div className="text-right">
                    <p className="text-xs text-zinc-500 uppercase tracking-wide">Date</p>
                    <p className="text-lg font-mono font-semibold text-white">
                      {state.currentDate ? formatDate(state.currentDate) : '-'}
                    </p>
                  </div>
                  <div className="text-right">
                    <p className="text-xs text-zinc-500 uppercase tracking-wide">Progress</p>
                    <p className="text-lg font-mono font-semibold text-white">
                      {state.progress.toFixed(1)}%
                    </p>
                  </div>
                  <div className="text-right">
                    <p className="text-xs text-zinc-500 uppercase tracking-wide">
                      SPY Benchmark
                    </p>
                    <p className={cn(
                      "text-lg font-mono font-semibold",
                      state.benchmarkReturn >= 0 ? "text-green-400" : "text-red-400"
                    )}>
                      {formatPercent(state.benchmarkReturn)}
                    </p>
                  </div>
                </div>
              </div>
              
              {/* Progress Bar */}
              {state.isRunning && (
                <div className="mt-4 h-1 bg-zinc-800 rounded-full overflow-hidden">
                  <div 
                    className="h-full bg-white transition-all duration-200"
                    style={{ width: `${state.progress}%` }}
                  />
                </div>
              )}
            </CardContent>
          </Card>

          {/* Main Content */}
          <div className="grid grid-cols-12 gap-4">
            {/* Leaderboard */}
            <div className="col-span-4">
              <Card className="h-[420px] bg-zinc-900 border-zinc-800">
                <CardHeader className="pb-3 border-b border-zinc-800">
                  <CardTitle className="text-sm font-medium text-zinc-300">
                    Fund Leaderboard
                  </CardTitle>
                </CardHeader>
                <CardContent className="p-3 overflow-auto h-[calc(100%-60px)]">
                  {state.leaderboard.length === 0 ? (
                    <div className="flex flex-col items-center justify-center h-full 
                                    text-zinc-500">
                      <p className="text-sm">Start simulation to see rankings</p>
                    </div>
                  ) : (
                    <div className="space-y-2">
                      {state.leaderboard.map((fund, index) => {
                        const theme = getFundTheme(fund.fund_id, funds);
                        const isSelected = selectedFund === fund.fund_id;
                        const isLeader = index === 0;
                        
                        return (
                          <div
                            key={fund.fund_id}
                            className={cn(
                              "p-3 rounded-lg cursor-pointer transition-all",
                              "border",
                              isSelected 
                                ? `bg-zinc-800 ${theme.border}` 
                                : "bg-zinc-900 border-zinc-800 hover:bg-zinc-800"
                            )}
                            onClick={() => setSelectedFund(fund.fund_id)}
                          >
                            <div className="flex items-center justify-between mb-2">
                              <div className="flex items-center gap-2">
                                <div className={cn(
                                  "w-6 h-6 rounded flex items-center justify-center",
                                  "text-sm font-bold",
                                  isLeader 
                                    ? "bg-zinc-700 text-white" 
                                    : "bg-zinc-800 text-zinc-400"
                                )}>
                                  {fund.rank}
                                </div>
                                <span className="font-medium text-white">
                                  {fund.fund_name}
                                </span>
                              </div>
                              <span className={cn(
                                "font-mono font-bold",
                                fund.cumulative_return >= 0 
                                  ? "text-green-400" 
                                  : "text-red-400"
                              )}>
                                {formatPercent(fund.cumulative_return)}
                              </span>
                            </div>
                            <div className="flex items-center justify-between text-xs">
                              <span className="text-zinc-400 font-mono">
                                {formatCurrency(fund.total_value)}
                              </span>
                              <div className="flex items-center gap-3 text-zinc-500">
                                <span>
                                  Sharpe <span className="text-zinc-300">
                                    {fund.sharpe_ratio.toFixed(2)}
                                  </span>
                                </span>
                                <span>
                                  DD <span className="text-orange-400">
                                    {(fund.max_drawdown * 100).toFixed(1)}%
                                  </span>
                                </span>
                              </div>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  )}
                </CardContent>
              </Card>
            </div>

            {/* Decision Stream */}
            <div className="col-span-4">
              <Card className="h-[420px] bg-zinc-900 border-zinc-800">
                <CardHeader className="pb-3 border-b border-zinc-800">
                  <CardTitle className="text-sm font-medium text-zinc-300">
                    Decision Stream
                  </CardTitle>
                </CardHeader>
                <CardContent className="p-3 overflow-auto h-[calc(100%-60px)]">
                  {state.decisions.length === 0 ? (
                    <div className="flex flex-col items-center justify-center h-full 
                                    text-zinc-500">
                      <p className="text-sm">AI fund managers are thinking...</p>
                    </div>
                  ) : (
                    <div className="space-y-2">
                      {state.decisions
                        .filter(d => !isParseError(d.reasoning))
                        .slice(0, 12)
                        .map((decision, i) => {
                          const theme = getFundTheme(decision.fund_id, funds);
                          const isBuy = decision.action.toLowerCase() === 'buy';
                          const isSell = decision.action.toLowerCase() === 'sell';
                          const isHold = decision.action.toLowerCase() === 'hold';
                          
                          return (
                            <div 
                              key={i} 
                              className={cn(
                                "p-3 rounded-lg border",
                                `bg-zinc-900 ${theme.border}`
                              )}
                            >
                              <div className="flex items-center justify-between mb-2">
                                <span className="font-medium text-white text-sm">
                                  {decision.fund_name}
                                </span>
                                <span className="text-xs text-zinc-500">
                                  {decision.simulation_date && 
                                    formatDate(decision.simulation_date)}
                                </span>
                              </div>
                              
                              <div className="flex items-center gap-2 mb-2">
                                <span className={cn(
                                  "px-2 py-0.5 rounded text-xs font-bold uppercase",
                                  isBuy && "bg-green-900 text-green-400",
                                  isSell && "bg-red-900 text-red-400",
                                  isHold && "bg-zinc-800 text-zinc-400"
                                )}>
                                  {decision.action}
                                </span>
                                {decision.symbol && (
                                  <span className="font-mono font-bold text-white">
                                    {decision.symbol}
                                  </span>
                                )}
                                {decision.confidence > 0 && (
                                  <span className="text-xs text-zinc-500 ml-auto">
                                    {(decision.confidence * 100).toFixed(0)}% conf
                                  </span>
                                )}
                              </div>
                              
                              <p className="text-xs text-zinc-400 leading-relaxed 
                                            line-clamp-2">
                                {decision.reasoning}
                              </p>
                            </div>
                          );
                        })}
                      <div ref={decisionsEndRef} />
                    </div>
                  )}
                </CardContent>
              </Card>
            </div>

            {/* Portfolio View */}
            <div className="col-span-4">
              <Card className="h-[420px] bg-zinc-900 border-zinc-800">
                <CardHeader className="pb-3 border-b border-zinc-800">
                  <CardTitle className="text-sm font-medium text-zinc-300">
                    {selectedPortfolio?.fund_name || 'Portfolio Details'}
                  </CardTitle>
                </CardHeader>
                <CardContent className="p-4 overflow-auto h-[calc(100%-60px)]">
                  {!selectedPortfolio ? (
                    <div className="flex flex-col items-center justify-center h-full 
                                    text-zinc-500">
                      <p className="text-sm">Select a fund to view details</p>
                    </div>
                  ) : (() => {
                    const theme = getFundTheme(selectedPortfolio.fund_id, funds);
                    return (
                      <div className="space-y-4">
                        {/* Stats Grid */}
                        <div className="grid grid-cols-2 gap-3">
                          <div className={cn(
                            "p-3 rounded-lg border",
                            `bg-zinc-900 ${theme.border}`
                          )}>
                            <p className="text-xs text-zinc-500 uppercase tracking-wide">
                              Total Value
                            </p>
                            <p className="text-xl font-bold text-white font-mono">
                              {formatCurrency(selectedPortfolio.total_value)}
                            </p>
                          </div>
                          <div className="p-3 rounded-lg bg-zinc-900 border border-zinc-800">
                            <p className="text-xs text-zinc-500 uppercase tracking-wide">
                              Cash
                            </p>
                            <p className="text-xl font-bold text-zinc-300 font-mono">
                              {formatCurrency(selectedPortfolio.cash)}
                            </p>
                          </div>
                          <div className="p-3 rounded-lg bg-zinc-900 border border-zinc-800">
                            <p className="text-xs text-zinc-500 uppercase tracking-wide">
                              Return
                            </p>
                            <p className={cn(
                              "text-xl font-bold font-mono",
                              selectedPortfolio.cumulative_return >= 0 
                                ? "text-green-400" 
                                : "text-red-400"
                            )}>
                              {formatPercent(selectedPortfolio.cumulative_return)}
                            </p>
                          </div>
                          <div className="p-3 rounded-lg bg-zinc-900 border border-zinc-800">
                            <p className="text-xs text-zinc-500 uppercase tracking-wide">
                              Max Drawdown
                            </p>
                            <p className="text-xl font-bold text-orange-400 font-mono">
                              {(selectedPortfolio.max_drawdown * 100).toFixed(1)}%
                            </p>
                          </div>
                        </div>
                        
                        {/* Positions */}
                        <div>
                          <p className="text-xs text-zinc-500 uppercase tracking-wide mb-2">
                            Positions
                          </p>
                          {Object.entries(selectedPortfolio.positions).length === 0 ? (
                            <div className="text-center py-4 text-zinc-600 text-sm">
                              No open positions
                            </div>
                          ) : (
                            <div className="space-y-1.5">
                              {Object.entries(selectedPortfolio.positions)
                                .sort((a, b) => b[1].current_value - a[1].current_value)
                                .map(([symbol, pos]) => (
                                  <div 
                                    key={symbol} 
                                    className="flex items-center justify-between p-2 
                                               rounded-lg bg-zinc-800"
                                  >
                                    <span className="font-mono font-bold text-white">
                                      {symbol}
                                    </span>
                                    <span className="text-xs text-zinc-500">
                                      {pos.quantity.toLocaleString()} @ 
                                      ${pos.current_price.toFixed(2)}
                                    </span>
                                    <span className="font-mono text-sm text-zinc-300">
                                      {formatCurrency(pos.current_value)}
                                    </span>
                                  </div>
                                ))}
                            </div>
                          )}
                        </div>
                      </div>
                    );
                  })()}
                </CardContent>
              </Card>
            </div>
          </div>

          {/* Debate Viewer */}
          <Card className="bg-zinc-900 border-zinc-800">
            <CardHeader className="pb-3 border-b border-zinc-800">
              <CardTitle className="text-sm font-medium text-zinc-300">
                AI Debate Log
              </CardTitle>
            </CardHeader>
            <CardContent className="p-4">
              {state.debates.length === 0 ? (
                <div className="text-center py-8 text-zinc-500">
                  <p className="text-sm">AI debate messages will appear here</p>
                </div>
              ) : (
                <div className="grid grid-cols-2 gap-3 max-h-[280px] overflow-auto">
                  {state.debates.slice(0, 8).map((msg, i) => {
                    const theme = getFundTheme(msg.fund_id, funds);
                    const phaseColors: Record<string, string> = {
                      'analyze': 'bg-blue-900 text-blue-400',
                      'propose': 'bg-green-900 text-green-400',
                      'decide': 'bg-amber-900 text-amber-400',
                      'confirm': 'bg-purple-900 text-purple-400',
                    };
                    const phaseColor = phaseColors[msg.phase] || 
                      'bg-zinc-800 text-zinc-400';
                    
                    return (
                      <div 
                        key={i} 
                        className={cn(
                          "p-3 rounded-lg border",
                          `bg-zinc-900 ${theme.border}`
                        )}
                      >
                        <div className="flex items-center gap-2 mb-2">
                          <span className={cn(
                            "px-2 py-0.5 rounded text-xs font-medium uppercase",
                            phaseColor
                          )}>
                            {msg.phase}
                          </span>
                          <span className="text-xs text-zinc-500 font-mono">
                            {msg.model}
                          </span>
                        </div>
                        <p className="text-xs text-zinc-400 leading-relaxed line-clamp-4">
                          {msg.content.slice(0, 300)}
                          {msg.content.length > 300 && '...'}
                        </p>
                      </div>
                    );
                  })}
                </div>
              )}
            </CardContent>
          </Card>
          
          {/* Export Section - Show when simulation is complete */}
          {!state.isRunning && state.simulationId && state.leaderboard.length > 0 && (
            <Card className="bg-zinc-900 border-zinc-800">
              <CardHeader className="pb-3 border-b border-zinc-800">
                <div className="flex items-center justify-between">
                  <CardTitle className="text-sm font-medium text-zinc-300">
                    Export Data
                  </CardTitle>
                  <span className="text-xs text-zinc-500">
                    Run ID: {state.simulationId}
                  </span>
                </div>
              </CardHeader>
              <CardContent className="p-4">
                <p className="text-sm text-zinc-400 mb-4">
                  Download simulation data for analysis or model training.
                </p>
                <div className="flex flex-wrap gap-2">
                  <a
                    href={`${API_BASE}/api/backtest/export/${state.simulationId}/trades.csv`}
                    download
                    className="px-3 py-2 bg-zinc-800 hover:bg-zinc-700 text-zinc-300 
                               text-sm rounded border border-zinc-700"
                  >
                    Download Trades (CSV)
                  </a>
                  <a
                    href={`${API_BASE}/api/backtest/export/${state.simulationId}/decisions.csv`}
                    download
                    className="px-3 py-2 bg-zinc-800 hover:bg-zinc-700 text-zinc-300 
                               text-sm rounded border border-zinc-700"
                  >
                    Download Decisions (CSV)
                  </a>
                  <a
                    href={`${API_BASE}/api/backtest/export/${state.simulationId}/snapshots.csv`}
                    download
                    className="px-3 py-2 bg-zinc-800 hover:bg-zinc-700 text-zinc-300 
                               text-sm rounded border border-zinc-700"
                  >
                    Download Snapshots (CSV)
                  </a>
                  <a
                    href={`${API_BASE}/api/backtest/export/${state.simulationId}/full.json`}
                    download
                    className="px-3 py-2 bg-white text-black hover:bg-zinc-200 
                               text-sm rounded font-medium"
                  >
                    Download Full Data (JSON)
                  </a>
                </div>
              </CardContent>
            </Card>
          )}
          
          {/* Historical Runs */}
          {historicalRuns.length > 0 && (
            <Card className="bg-zinc-900 border-zinc-800">
              <CardHeader className="pb-3 border-b border-zinc-800">
                <div className="flex items-center justify-between">
                  <CardTitle className="text-sm font-medium text-zinc-300">
                    Previous Runs ({historicalRuns.length})
                  </CardTitle>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => setShowHistory(!showHistory)}
                    className="text-zinc-400 hover:text-white"
                  >
                    {showHistory ? 'Hide' : 'Show'}
                  </Button>
                </div>
              </CardHeader>
              {showHistory && (
                <CardContent className="p-4">
                  <div className="space-y-2 max-h-[300px] overflow-auto">
                    {historicalRuns.slice(0, 10).map((run) => (
                      <div 
                        key={run.id}
                        className="p-3 bg-zinc-800/50 rounded-lg border border-zinc-700/50"
                      >
                        <div className="flex items-center justify-between mb-2">
                          <span className="font-mono text-sm text-white">
                            {run.id}
                          </span>
                          <span className={cn(
                            "text-xs px-2 py-0.5 rounded",
                            run.status === 'completed' 
                              ? "bg-green-900 text-green-400"
                              : run.status === 'failed'
                              ? "bg-red-900 text-red-400"
                              : "bg-zinc-700 text-zinc-400"
                          )}>
                            {run.status}
                          </span>
                        </div>
                        <div className="flex items-center justify-between text-xs 
                                        text-zinc-500">
                          <span>
                            {run.total_trades || 0} trades, 
                            {run.total_decisions || 0} decisions
                          </span>
                          <span>
                            {run.created_at ? 
                              new Date(run.created_at).toLocaleDateString() : '-'}
                          </span>
                        </div>
                        {run.status === 'completed' && (
                          <div className="flex gap-2 mt-2">
                            <a
                              href={`${API_BASE}/api/backtest/export/${run.id}/full.json`}
                              download
                              className="text-xs text-blue-400 hover:text-blue-300"
                            >
                              Export JSON
                            </a>
                            <a
                              href={`${API_BASE}/api/backtest/export/${run.id}/trades.csv`}
                              download
                              className="text-xs text-blue-400 hover:text-blue-300"
                            >
                              Export CSV
                            </a>
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                  <p className="text-xs text-zinc-600 mt-3">
                    Use this data to train models for better performance in future runs.
                  </p>
                </CardContent>
              )}
            </Card>
          )}
        </>
      )}
      </div>
    </div>
  );
}
