// Manager types
export type ManagerType = 'llm' | 'quant';
export type ManagerProvider = 'openai' | 'anthropic' | 'google' | null;

export interface Manager {
  id: string;
  name: string;
  type: ManagerType;
  provider: ManagerProvider;
  isActive: boolean;
  description?: string;
}

// Portfolio types
export interface Portfolio {
  managerId: string;
  cashBalance: number;
  totalValue: number;
  updatedAt: string;
}

// Position types
export interface Position {
  id: number;
  managerId: string;
  symbol: string;
  quantity: number;
  avgEntryPrice: number;
  currentPrice: number;
  unrealizedPnl: number;
  openedAt: string;
}

// Trade types
export type TradeSide = 'buy' | 'sell';

export interface Trade {
  id: number;
  managerId: string;
  symbol: string;
  side: TradeSide;
  quantity: number;
  price: number;
  reasoning?: string;
  signalsUsed?: Record<string, unknown>;
  executedAt: string;
}

// Performance snapshot types
export interface DailySnapshot {
  id: number;
  managerId: string;
  date: string;
  portfolioValue: number;
  dailyReturn: number;
  cumulativeReturn: number;
  sharpeRatio: number;
}

// Strategy signal types
export interface MomentumSignal {
  symbol: string;
  score: number; // -1 to +1
}

export interface MeanReversionSignal {
  symbol: string;
  score: number; // -1 to +1
}

export interface TechnicalIndicators {
  symbol: string;
  rsi: number;
  macd: {
    macd: number;
    signal: number;
    histogram: number;
  };
  sma20: number;
  sma50: number;
  sma200: number;
  atr: number;
}

export interface MLPrediction {
  symbol: string;
  predictedReturn: number;
  confidence: number;
}

export type VolatilityRegime = 
  | 'low_vol_trending_up'
  | 'low_vol_trending_down'
  | 'low_vol_ranging'
  | 'normal_vol_trending_up'
  | 'normal_vol_trending_down'
  | 'normal_vol_ranging'
  | 'high_vol_trending_up'
  | 'high_vol_trending_down'
  | 'high_vol_ranging';

export interface SemanticSearchResult {
  similarPeriods: Array<{
    date: string;
    similarity: number;
    return5d: number;
    return20d: number;
  }>;
  avg5dReturn: number;
  avg20dReturn: number;
  positive5dRate: number;
  interpretation: string;
}

export interface StrategySignals {
  momentum: MomentumSignal[];
  meanReversion: MeanReversionSignal[];
  technical: TechnicalIndicators[];
  mlPrediction: MLPrediction[];
  volatilityRegime: VolatilityRegime;
  semanticSearch: SemanticSearchResult;
  timestamp: string;
}

// Leaderboard types
export interface LeaderboardEntry {
  rank: number;
  manager: Manager;
  portfolio: Portfolio;
  sharpeRatio: number;
  totalReturn: number;
  volatility: number;
  maxDrawdown: number;
  totalTrades: number;
  winRate: number;
}

// Market data types
export interface OHLCV {
  symbol: string;
  timestamp: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

// API response types
export interface ApiResponse<T> {
  success: boolean;
  data?: T;
  error?: string;
}
