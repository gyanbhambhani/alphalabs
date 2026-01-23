import { create } from 'zustand';
import type { 
  Manager, 
  Portfolio, 
  LeaderboardEntry, 
  StrategySignals,
  Trade 
} from '@/types';

interface TradingState {
  // Data
  managers: Manager[];
  portfolios: Record<string, Portfolio>;
  leaderboard: LeaderboardEntry[];
  signals: StrategySignals | null;
  recentTrades: Trade[];
  
  // Loading states
  isLoading: boolean;
  error: string | null;
  
  // Selected manager for detail view
  selectedManagerId: string | null;
  
  // Actions
  setManagers: (managers: Manager[]) => void;
  setPortfolios: (portfolios: Portfolio[]) => void;
  setLeaderboard: (leaderboard: LeaderboardEntry[]) => void;
  setSignals: (signals: StrategySignals) => void;
  setRecentTrades: (trades: Trade[]) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  selectManager: (id: string | null) => void;
}

export const useTradingStore = create<TradingState>((set) => ({
  // Initial data
  managers: [],
  portfolios: {},
  leaderboard: [],
  signals: null,
  recentTrades: [],
  
  // Initial loading state
  isLoading: false,
  error: null,
  
  // Initial selection
  selectedManagerId: null,
  
  // Actions
  setManagers: (managers) => set({ managers }),
  
  setPortfolios: (portfolios) => {
    const portfolioMap: Record<string, Portfolio> = {};
    portfolios.forEach((p) => {
      portfolioMap[p.managerId] = p;
    });
    set({ portfolios: portfolioMap });
  },
  
  setLeaderboard: (leaderboard) => set({ leaderboard }),
  
  setSignals: (signals) => set({ signals }),
  
  setRecentTrades: (trades) => set({ recentTrades: trades }),
  
  setLoading: (isLoading) => set({ isLoading }),
  
  setError: (error) => set({ error }),
  
  selectManager: (selectedManagerId) => set({ selectedManagerId }),
}));
