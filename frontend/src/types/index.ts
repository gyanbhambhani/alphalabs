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
  | 'high_vol_ranging'
  | 'unknown'
  | string;  // Allow any string for flexibility

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
  dataFreshness?: string;  // "live", "recent", "X days old", or "error"
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

// Embedding types
export interface EmbeddingMetadata {
  date: string;
  return1w: number;
  return1m: number;
  return3m: number;
  return6m: number;
  return12m: number;
  volatility5d: number;
  volatility10d: number;
  volatility21d: number;
  volatility63d: number;
  price: number;
}

export interface Embedding {
  id: string;
  metadata: EmbeddingMetadata;
}

export interface EmbeddingSearchResult extends Embedding {
  similarity: number;
  queryInterpretation?: string;
}

export interface EmbeddingsStats {
  totalCount: number;
  dateRange: [string, string];
  avgReturn1m: number;
  avgVolatility21d: number;
}

export interface EmbeddingsListResponse {
  embeddings: Embedding[];
  total: number;
  page: number;
  perPage: number;
}

export interface EmbeddingSearchResponse {
  results: EmbeddingSearchResult[];
  interpretation: string;
  total: number;
}

// Stock types
export interface Stock {
  symbol: string;
  name: string;
  sector: string | null;
  subIndustry: string | null;
  headquarters: string | null;
  hasEmbeddings: boolean;
  embeddingsCount: number;
  embeddingsDateRangeStart: string | null;
  embeddingsDateRangeEnd: string | null;
}

export interface StocksListResponse {
  stocks: Stock[];
  total: number;
  sectors: string[];
}

// =============================================================================
// Collaborative Fund Types
// =============================================================================

export type FundStrategy = 
  | 'trend_macro'
  | 'mean_reversion'
  | 'event_driven'
  | 'quality_ls';

export interface Fund {
  fundId: string;
  name: string;
  strategy: FundStrategy;
  description?: string;
  totalValue: number;
  cashBalance: number;
  grossExposure: number;
  netExposure: number;
  nPositions: number;
  isActive: boolean;
}

export interface FundDetail extends Fund {
  thesis?: FundThesis;
  policy?: FundPolicy;
  riskLimits?: FundRiskLimits;
}

export interface FundThesis {
  name: string;
  strategy: string;
  description: string;
  horizonDays: [number, number];
  universeSpec: {
    type: string;
    params: Record<string, unknown>;
  };
  edge: string;
  version: string;
}

export interface FundPolicy {
  sizingMethod: string;
  volTarget?: number;
  maxPositionPct: number;
  maxTurnoverDaily: number;
  rebalanceCadence: string;
  maxPositions: number;
  defaultStopLossPct: number;
  defaultTakeProfitPct: number;
  trailingStop: boolean;
  maxGrossExposure: number;
  minCashBuffer: number;
  goFlatOnCircuitBreaker: boolean;
  version: string;
}

export interface FundRiskLimits {
  maxPositionPct: number;
  maxSectorPct: number;
  maxGrossExposure: number;
  maxDailyLossPct: number;
  maxWeeklyDrawdownPct: number;
  breachAction: string;
  breachCooldownDays: number;
}

export interface FundPosition {
  symbol: string;
  quantity: number;
  avgEntryPrice: number;
  currentPrice: number;
  marketValue: number;
  unrealizedPnl: number;
  weightPct: number;
}

export type DecisionType = 'trade' | 'no_trade';
export type DecisionStatus = 
  | 'created'
  | 'debated'
  | 'risk_vetoed'
  | 'finalized'
  | 'sent_to_broker'
  | 'partially_filled'
  | 'filled'
  | 'canceled'
  | 'errored';

export type NoTradeReason = 
  | 'snapshot_invalid'
  | 'risk_veto'
  | 'disagreement'
  | 'baseline_failed'
  | 'cooldown'
  | 'universe_empty'
  | 'no_opportunities';

export interface DecisionRecord {
  decisionId: string;
  fundId: string;
  snapshotId: string;
  asofTimestamp: string;
  decisionType: DecisionType;
  status: DecisionStatus;
  noTradeReason?: NoTradeReason;
  universeHash?: string;
  inputsHash?: string;
  predictedDirections?: Record<string, string>;
  expectedReturn?: number;
  expectedHoldingDays?: number;
}

export interface DecisionDetail extends DecisionRecord {
  idempotencyKey: string;
  runContext: string;
  statusHistory: StatusTransition[];
  intent?: PortfolioIntent;
  riskResult?: RiskCheckResult;
  snapshotQuality?: Record<string, unknown>;
  universeResult?: Record<string, unknown>;
  modelVersions?: Record<string, string>;
  promptHashes?: Record<string, string>;
}

export interface StatusTransition {
  fromStatus: DecisionStatus;
  toStatus: DecisionStatus;
  timestamp: string;
  reason: string;
}

export interface PortfolioIntent {
  intentId: string;
  fundId: string;
  asofTimestamp: string;
  portfolioValue: number;
  positions: PositionIntent[];
  targetCashPct: number;
  sizingMethodUsed: string;
  policyVersion: string;
}

export interface PositionIntent {
  symbol: string;
  targetWeight: number;
  direction: string;
  urgency: string;
}

export interface RiskCheckResult {
  status: string;
  scaleFactor: number;
  perSymbolScales: Record<string, number>;
  violations: RiskViolation[];
  appliedRules: string[];
  reason?: string;
}

export interface RiskViolation {
  ruleName: string;
  symbol?: string;
  limit: number;
  actual: number;
  severity: string;
}

export interface DebateTranscript {
  transcriptId: string;
  fundId: string;
  snapshotId: string;
  startedAt: string;
  completedAt?: string;
  numProposals: number;
  numCritiques: number;
  finalConsensusLevel: number;
  messages?: DebateMessage[];
  totalInputTokens?: number;
  totalOutputTokens?: number;
}

export interface DebateMessage {
  phase: string;
  participantId: string;
  timestamp: string;
  content: Record<string, unknown>;
  modelName: string;
  modelVersion: string;
  promptHash: string;
}

export interface FundLeaderboardEntry {
  rank: number;
  fundId: string;
  name: string;
  strategy: FundStrategy;
  totalValue: number;
  grossExposure: number;
  isActive: boolean;
}
