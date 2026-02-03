"""
Backtest Persistence Service.

Saves trades, decisions, and snapshots to SQLite for:
- Reviewing historical trades after simulation ends
- Comparing different simulation runs
- Training models on historical decisions
- Exporting trade data (CSV/JSON)
"""

import uuid
import json
import csv
import io
from datetime import date, datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import logging

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session

logger = logging.getLogger(__name__)

# Import models - we'll create SQLite-specific versions
from db.models import (
    Base,
    BacktestRun,
    BacktestTradeRecord,
    BacktestDecisionRecord,
    BacktestPortfolioSnapshotRecord,
)


class BacktestPersistence:
    """
    Handles all database operations for backtest data.
    
    Uses SQLite for simplicity and portability.
    """
    
    def __init__(self, db_path: str = "./data/backtest_history.db"):
        """
        Initialize the persistence layer.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create SQLite engine
        self.engine = create_engine(
            f"sqlite:///{self.db_path}",
            echo=False,
            connect_args={"check_same_thread": False},
        )
        
        # Create session factory
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine,
        )
        
        # Initialize tables
        self._init_tables()
        
        logger.info(f"Backtest persistence initialized at {self.db_path}")
    
    def _init_tables(self):
        """Create all backtest tables if they don't exist."""
        # Only create the backtest-related tables
        BacktestRun.__table__.create(self.engine, checkfirst=True)
        BacktestTradeRecord.__table__.create(self.engine, checkfirst=True)
        BacktestDecisionRecord.__table__.create(self.engine, checkfirst=True)
        BacktestPortfolioSnapshotRecord.__table__.create(self.engine, checkfirst=True)
        
        # Create fund metrics table
        self._ensure_metrics_table()
        
        logger.info("Backtest tables initialized")
    
    def get_session(self) -> Session:
        """Get a new database session."""
        return self.SessionLocal()
    
    # =========================================================================
    # Run Management
    # =========================================================================
    
    def create_run(
        self,
        run_id: str,
        fund_ids: List[str],
        start_date: date,
        end_date: date,
        initial_cash: float,
        config: Optional[Dict] = None,
        universe: Optional[List[str]] = None,
    ) -> BacktestRun:
        """Create a new backtest run record."""
        with self.get_session() as session:
            run = BacktestRun(
                id=run_id,
                start_date=start_date,
                end_date=end_date,
                fund_ids=fund_ids,
                initial_cash=initial_cash,
                config=config,
                universe=universe,
                status="running",
                created_at=datetime.utcnow(),
                started_at=datetime.utcnow(),
            )
            session.add(run)
            session.commit()
            session.refresh(run)
            logger.info(f"Created backtest run: {run_id}")
            return run
    
    def update_run_progress(
        self,
        run_id: str,
        current_day: int,
        total_days: int,
    ):
        """Update run progress."""
        with self.get_session() as session:
            session.execute(
                text("""
                    UPDATE backtest_runs 
                    SET current_day = :current_day, total_days = :total_days
                    WHERE id = :run_id
                """),
                {"run_id": run_id, "current_day": current_day, "total_days": total_days}
            )
            session.commit()
    
    def complete_run(
        self,
        run_id: str,
        total_trades: int,
        total_decisions: int,
        elapsed_seconds: float,
    ):
        """Mark a run as completed."""
        with self.get_session() as session:
            session.execute(
                text("""
                    UPDATE backtest_runs 
                    SET status = 'completed',
                        completed_at = :completed_at,
                        total_trades = :total_trades,
                        total_decisions = :total_decisions,
                        elapsed_seconds = :elapsed_seconds
                    WHERE id = :run_id
                """),
                {
                    "run_id": run_id,
                    "completed_at": datetime.utcnow(),
                    "total_trades": total_trades,
                    "total_decisions": total_decisions,
                    "elapsed_seconds": elapsed_seconds,
                }
            )
            session.commit()
            logger.info(f"Completed backtest run: {run_id}")
    
    def fail_run(self, run_id: str, error_message: str):
        """Mark a run as failed."""
        with self.get_session() as session:
            session.execute(
                text("""
                    UPDATE backtest_runs 
                    SET status = 'failed', error_message = :error_message
                    WHERE id = :run_id
                """),
                {"run_id": run_id, "error_message": error_message}
            )
            session.commit()
    
    # =========================================================================
    # Trade Recording
    # =========================================================================
    
    def save_trade(
        self,
        run_id: str,
        fund_id: str,
        trade_date: date,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        commission: float = 0.0,
        reasoning: Optional[str] = None,
        confidence: Optional[float] = None,
        signals_snapshot: Optional[Dict] = None,
    ) -> str:
        """Save a trade record."""
        trade_id = str(uuid.uuid4())[:12]
        
        with self.get_session() as session:
            trade = BacktestTradeRecord(
                id=trade_id,
                run_id=run_id,
                fund_id=fund_id,
                trade_date=trade_date,
                symbol=symbol,
                side=side,
                quantity=quantity,
                price=price,
                commission=commission,
                total_cost=quantity * price + (commission if side == 'buy' else -commission),
                reasoning=reasoning,
                confidence=confidence,
                signals_snapshot=signals_snapshot,
                executed_at=datetime.utcnow(),
                created_at=datetime.utcnow(),
            )
            session.add(trade)
            session.commit()
            
        return trade_id
    
    def save_trades_batch(self, trades: List[Dict]):
        """Save multiple trades at once for efficiency."""
        if not trades:
            return
            
        with self.get_session() as session:
            for trade_data in trades:
                trade = BacktestTradeRecord(
                    id=str(uuid.uuid4())[:12],
                    created_at=datetime.utcnow(),
                    executed_at=datetime.utcnow(),
                    **trade_data
                )
                session.add(trade)
            session.commit()
            logger.debug(f"Saved batch of {len(trades)} trades")
    
    # =========================================================================
    # Decision Recording
    # =========================================================================
    
    def save_decision(
        self,
        run_id: str,
        fund_id: str,
        decision_date: date,
        action: str,
        symbol: Optional[str] = None,
        quantity: Optional[float] = None,
        target_weight: Optional[float] = None,
        confidence: Optional[float] = None,
        reasoning: Optional[str] = None,
        debate_transcript: Optional[List[Dict]] = None,
        signals_snapshot: Optional[Dict] = None,
        models_used: Optional[Dict] = None,
        tokens_used: int = 0,
        triggered_by: Optional[str] = None,
    ) -> str:
        """Save a decision record (including HOLDs)."""
        decision_id = str(uuid.uuid4())[:12]
        
        with self.get_session() as session:
            decision = BacktestDecisionRecord(
                id=decision_id,
                run_id=run_id,
                fund_id=fund_id,
                decision_date=decision_date,
                action=action,
                symbol=symbol,
                quantity=quantity,
                target_weight=target_weight,
                confidence=confidence,
                reasoning=reasoning,
                debate_transcript=debate_transcript,
                signals_snapshot=signals_snapshot,
                models_used=models_used,
                tokens_used=tokens_used,
                triggered_by=triggered_by,
                created_at=datetime.utcnow(),
            )
            session.add(decision)
            session.commit()
            
        return decision_id
    
    # =========================================================================
    # Snapshot Recording
    # =========================================================================
    
    def save_portfolio_snapshot(
        self,
        run_id: str,
        fund_id: str,
        snapshot_date: date,
        cash: float,
        positions: Dict,
        total_value: float,
        daily_return: Optional[float] = None,
        cumulative_return: Optional[float] = None,
        max_drawdown: Optional[float] = None,
        sharpe_ratio: Optional[float] = None,
        n_trades_today: int = 0,
    ) -> str:
        """Save a daily portfolio snapshot."""
        snapshot_id = str(uuid.uuid4())[:12]
        
        with self.get_session() as session:
            snapshot = BacktestPortfolioSnapshotRecord(
                id=snapshot_id,
                run_id=run_id,
                fund_id=fund_id,
                snapshot_date=snapshot_date,
                cash=cash,
                positions=positions,
                total_value=total_value,
                invested_pct=1.0 - (cash / total_value) if total_value > 0 else 0,
                daily_return=daily_return,
                cumulative_return=cumulative_return,
                max_drawdown=max_drawdown,
                sharpe_ratio=sharpe_ratio,
                n_positions=len(positions) if positions else 0,
                n_trades_today=n_trades_today,
                created_at=datetime.utcnow(),
            )
            session.add(snapshot)
            session.commit()
            
        return snapshot_id
    
    # =========================================================================
    # Query Methods
    # =========================================================================
    
    def get_all_runs(self) -> List[Dict]:
        """Get all backtest runs."""
        with self.get_session() as session:
            result = session.execute(
                text("""
                    SELECT id, name, start_date, end_date, status, 
                           total_trades, total_decisions, elapsed_seconds,
                           created_at, completed_at, fund_ids, initial_cash
                    FROM backtest_runs
                    ORDER BY created_at DESC
                """)
            )
            runs = []
            for row in result:
                runs.append({
                    "id": row[0],
                    "name": row[1],
                    "start_date": str(row[2]) if row[2] else None,
                    "end_date": str(row[3]) if row[3] else None,
                    "status": row[4],
                    "total_trades": row[5],
                    "total_decisions": row[6],
                    "elapsed_seconds": row[7],
                    "created_at": str(row[8]) if row[8] else None,
                    "completed_at": str(row[9]) if row[9] else None,
                    "fund_ids": json.loads(row[10]) if row[10] else [],
                    "initial_cash": row[11],
                })
            return runs
    
    def get_run(self, run_id: str) -> Optional[Dict]:
        """Get a specific run with all details."""
        with self.get_session() as session:
            result = session.execute(
                text("SELECT * FROM backtest_runs WHERE id = :run_id"),
                {"run_id": run_id}
            ).fetchone()
            
            if not result:
                return None
                
            return dict(result._mapping)
    
    def get_trades_for_run(
        self, 
        run_id: str, 
        fund_id: Optional[str] = None
    ) -> List[Dict]:
        """Get all trades for a run, optionally filtered by fund."""
        with self.get_session() as session:
            if fund_id:
                result = session.execute(
                    text("""
                        SELECT * FROM backtest_trades 
                        WHERE run_id = :run_id AND fund_id = :fund_id
                        ORDER BY trade_date, executed_at
                    """),
                    {"run_id": run_id, "fund_id": fund_id}
                )
            else:
                result = session.execute(
                    text("""
                        SELECT * FROM backtest_trades 
                        WHERE run_id = :run_id
                        ORDER BY trade_date, executed_at
                    """),
                    {"run_id": run_id}
                )
            
            return [dict(row._mapping) for row in result]
    
    def get_decisions_for_run(
        self, 
        run_id: str, 
        fund_id: Optional[str] = None,
        action: Optional[str] = None,
    ) -> List[Dict]:
        """Get all decisions for a run."""
        with self.get_session() as session:
            query = "SELECT * FROM backtest_decisions WHERE run_id = :run_id"
            params = {"run_id": run_id}
            
            if fund_id:
                query += " AND fund_id = :fund_id"
                params["fund_id"] = fund_id
            
            if action:
                query += " AND action = :action"
                params["action"] = action
            
            query += " ORDER BY decision_date"
            
            result = session.execute(text(query), params)
            return [dict(row._mapping) for row in result]
    
    def get_decision_with_transcript(
        self,
        decision_id: str,
    ) -> Optional[Dict]:
        """
        Get a single decision with its full debate transcript.
        
        Returns the decision including all 4 phases:
        - analyze: Market analysis (Gemini)
        - propose: Trade proposal (GPT)
        - decide: Final decision (GPT)
        - confirm: Risk confirmation (Claude) - if applicable
        """
        with self.get_session() as session:
            result = session.execute(
                text("SELECT * FROM backtest_decisions WHERE id = :decision_id"),
                {"decision_id": decision_id}
            ).fetchone()
            
            if not result:
                return None
            
            decision = dict(result._mapping)
            
            # Parse debate_transcript JSON if it's a string
            if decision.get("debate_transcript"):
                if isinstance(decision["debate_transcript"], str):
                    decision["debate_transcript"] = json.loads(
                        decision["debate_transcript"]
                    )
            
            return decision
    
    def get_debate_transcripts_for_run(
        self,
        run_id: str,
        fund_id: Optional[str] = None,
    ) -> List[Dict]:
        """
        Get all debate transcripts for a run.
        
        Returns structured data with all 4 phases for each decision.
        """
        with self.get_session() as session:
            query = """
                SELECT id, fund_id, decision_date, action, symbol, confidence,
                       debate_transcript, models_used, tokens_used
                FROM backtest_decisions
                WHERE run_id = :run_id AND debate_transcript IS NOT NULL
            """
            params = {"run_id": run_id}
            
            if fund_id:
                query += " AND fund_id = :fund_id"
                params["fund_id"] = fund_id
            
            query += " ORDER BY decision_date"
            
            result = session.execute(text(query), params)
            
            transcripts = []
            for row in result:
                row_dict = dict(row._mapping)
                
                # Parse JSON
                if row_dict.get("debate_transcript"):
                    if isinstance(row_dict["debate_transcript"], str):
                        row_dict["debate_transcript"] = json.loads(
                            row_dict["debate_transcript"]
                        )
                
                if row_dict.get("models_used"):
                    if isinstance(row_dict["models_used"], str):
                        row_dict["models_used"] = json.loads(
                            row_dict["models_used"]
                        )
                
                transcripts.append(row_dict)
            
            return transcripts
    
    def get_snapshots_for_run(
        self, 
        run_id: str, 
        fund_id: Optional[str] = None
    ) -> List[Dict]:
        """Get all portfolio snapshots for a run."""
        with self.get_session() as session:
            if fund_id:
                result = session.execute(
                    text("""
                        SELECT * FROM backtest_portfolio_snapshots 
                        WHERE run_id = :run_id AND fund_id = :fund_id
                        ORDER BY snapshot_date
                    """),
                    {"run_id": run_id, "fund_id": fund_id}
                )
            else:
                result = session.execute(
                    text("""
                        SELECT * FROM backtest_portfolio_snapshots 
                        WHERE run_id = :run_id
                        ORDER BY fund_id, snapshot_date
                    """),
                    {"run_id": run_id}
                )
            
            return [dict(row._mapping) for row in result]
    
    # =========================================================================
    # Export Methods
    # =========================================================================
    
    def export_trades_csv(self, run_id: str) -> str:
        """Export trades to CSV string."""
        trades = self.get_trades_for_run(run_id)
        
        if not trades:
            return ""
        
        output = io.StringIO()
        writer = csv.DictWriter(
            output, 
            fieldnames=[
                'trade_date', 'fund_id', 'symbol', 'side', 
                'quantity', 'price', 'commission', 'total_cost',
                'reasoning', 'confidence'
            ]
        )
        writer.writeheader()
        
        for trade in trades:
            writer.writerow({
                'trade_date': trade.get('trade_date'),
                'fund_id': trade.get('fund_id'),
                'symbol': trade.get('symbol'),
                'side': trade.get('side'),
                'quantity': trade.get('quantity'),
                'price': trade.get('price'),
                'commission': trade.get('commission'),
                'total_cost': trade.get('total_cost'),
                'reasoning': trade.get('reasoning', '')[:200] if trade.get('reasoning') else '',
                'confidence': trade.get('confidence'),
            })
        
        return output.getvalue()
    
    def export_decisions_csv(self, run_id: str) -> str:
        """Export decisions to CSV string."""
        decisions = self.get_decisions_for_run(run_id)
        
        if not decisions:
            return ""
        
        output = io.StringIO()
        writer = csv.DictWriter(
            output, 
            fieldnames=[
                'decision_date', 'fund_id', 'action', 'symbol', 
                'quantity', 'target_weight', 'confidence', 'reasoning',
                'tokens_used', 'triggered_by'
            ]
        )
        writer.writeheader()
        
        for decision in decisions:
            writer.writerow({
                'decision_date': decision.get('decision_date'),
                'fund_id': decision.get('fund_id'),
                'action': decision.get('action'),
                'symbol': decision.get('symbol'),
                'quantity': decision.get('quantity'),
                'target_weight': decision.get('target_weight'),
                'confidence': decision.get('confidence'),
                'reasoning': decision.get('reasoning', '')[:200] if decision.get('reasoning') else '',
                'tokens_used': decision.get('tokens_used'),
                'triggered_by': decision.get('triggered_by'),
            })
        
        return output.getvalue()
    
    def export_snapshots_csv(self, run_id: str) -> str:
        """Export portfolio snapshots to CSV string."""
        snapshots = self.get_snapshots_for_run(run_id)
        
        if not snapshots:
            return ""
        
        output = io.StringIO()
        writer = csv.DictWriter(
            output, 
            fieldnames=[
                'snapshot_date', 'fund_id', 'cash', 'total_value',
                'invested_pct', 'daily_return', 'cumulative_return',
                'max_drawdown', 'sharpe_ratio', 'n_positions', 'n_trades_today'
            ]
        )
        writer.writeheader()
        
        for snapshot in snapshots:
            writer.writerow({
                'snapshot_date': snapshot.get('snapshot_date'),
                'fund_id': snapshot.get('fund_id'),
                'cash': snapshot.get('cash'),
                'total_value': snapshot.get('total_value'),
                'invested_pct': snapshot.get('invested_pct'),
                'daily_return': snapshot.get('daily_return'),
                'cumulative_return': snapshot.get('cumulative_return'),
                'max_drawdown': snapshot.get('max_drawdown'),
                'sharpe_ratio': snapshot.get('sharpe_ratio'),
                'n_positions': snapshot.get('n_positions'),
                'n_trades_today': snapshot.get('n_trades_today'),
            })
        
        return output.getvalue()
    
    def export_run_json(self, run_id: str) -> Dict:
        """Export complete run data as JSON."""
        run = self.get_run(run_id)
        if not run:
            return {}
        
        return {
            "run": run,
            "trades": self.get_trades_for_run(run_id),
            "decisions": self.get_decisions_for_run(run_id),
            "snapshots": self.get_snapshots_for_run(run_id),
        }
    
    # =========================================================================
    # Training Data Export
    # =========================================================================
    
    def export_training_data(self, run_id: Optional[str] = None) -> List[Dict]:
        """
        Export data formatted for training ML models.
        
        Each record contains:
        - Market state (signals snapshot)
        - Decision made (action, symbol, confidence)
        - Outcome (did the trade profit?)
        
        This can be used to train models to make better decisions.
        """
        with self.get_session() as session:
            if run_id:
                decisions = session.execute(
                    text("""
                        SELECT d.*, 
                               t.price as trade_price,
                               t.quantity as trade_quantity
                        FROM backtest_decisions d
                        LEFT JOIN backtest_trades t 
                            ON d.run_id = t.run_id 
                            AND d.fund_id = t.fund_id 
                            AND d.symbol = t.symbol
                            AND d.decision_date = t.trade_date
                        WHERE d.run_id = :run_id
                        ORDER BY d.decision_date
                    """),
                    {"run_id": run_id}
                )
            else:
                # Get from all runs
                decisions = session.execute(
                    text("""
                        SELECT d.*, 
                               t.price as trade_price,
                               t.quantity as trade_quantity
                        FROM backtest_decisions d
                        LEFT JOIN backtest_trades t 
                            ON d.run_id = t.run_id 
                            AND d.fund_id = t.fund_id 
                            AND d.symbol = t.symbol
                            AND d.decision_date = t.trade_date
                        ORDER BY d.decision_date
                    """)
                )
            
            training_data = []
            for row in decisions:
                row_dict = dict(row._mapping)
                training_data.append({
                    "date": str(row_dict.get('decision_date')),
                    "fund_id": row_dict.get('fund_id'),
                    "action": row_dict.get('action'),
                    "symbol": row_dict.get('symbol'),
                    "confidence": row_dict.get('confidence'),
                    "reasoning": row_dict.get('reasoning'),
                    "signals": row_dict.get('signals_snapshot'),
                    "trade_price": row_dict.get('trade_price'),
                    "triggered_by": row_dict.get('triggered_by'),
                })
            
            return training_data
    
    # =========================================================================
    # Fund Metrics Recording
    # =========================================================================
    
    def save_fund_metrics(
        self,
        run_id: str,
        fund_id: str,
        metrics: Dict,
    ) -> None:
        """
        Save computed fund metrics for a run.
        
        Args:
            run_id: Backtest run ID
            fund_id: Fund identifier
            metrics: Dictionary of FundMetrics fields
        """
        with self.get_session() as session:
            # Store metrics as JSON in a dedicated field or separate table
            # For simplicity, we'll add to the run record
            session.execute(
                text("""
                    INSERT OR REPLACE INTO backtest_fund_metrics 
                    (run_id, fund_id, period_start, period_end, n_trades,
                     n_winning_trades, n_losing_trades, avg_return, median_return,
                     max_return, min_return, hit_rate, brier_score, turnover,
                     avg_holding_days, avg_slippage_bps, max_drawdown, created_at)
                    VALUES (:run_id, :fund_id, :period_start, :period_end, :n_trades,
                            :n_winning_trades, :n_losing_trades, :avg_return, 
                            :median_return, :max_return, :min_return, :hit_rate, 
                            :brier_score, :turnover, :avg_holding_days, 
                            :avg_slippage_bps, :max_drawdown, :created_at)
                """),
                {
                    "run_id": run_id,
                    "fund_id": fund_id,
                    "period_start": str(metrics.get("period_start")),
                    "period_end": str(metrics.get("period_end")),
                    "n_trades": metrics.get("n_trades", 0),
                    "n_winning_trades": metrics.get("n_winning_trades", 0),
                    "n_losing_trades": metrics.get("n_losing_trades", 0),
                    "avg_return": metrics.get("avg_return", 0.0),
                    "median_return": metrics.get("median_return", 0.0),
                    "max_return": metrics.get("max_return", 0.0),
                    "min_return": metrics.get("min_return", 0.0),
                    "hit_rate": metrics.get("hit_rate", 0.0),
                    "brier_score": metrics.get("brier_score", 0.0),
                    "turnover": metrics.get("turnover", 0.0),
                    "avg_holding_days": metrics.get("avg_holding_days", 0.0),
                    "avg_slippage_bps": metrics.get("avg_slippage_bps", 0.0),
                    "max_drawdown": metrics.get("max_drawdown", 0.0),
                    "created_at": datetime.utcnow(),
                }
            )
            session.commit()
            logger.info(f"Saved fund metrics for {fund_id} in run {run_id}")
    
    def get_fund_metrics(
        self, 
        run_id: str, 
        fund_id: Optional[str] = None
    ) -> List[Dict]:
        """
        Get fund metrics for a run.
        
        Args:
            run_id: Backtest run ID
            fund_id: Optional fund filter
            
        Returns:
            List of fund metrics dictionaries
        """
        with self.get_session() as session:
            try:
                if fund_id:
                    result = session.execute(
                        text("""
                            SELECT * FROM backtest_fund_metrics 
                            WHERE run_id = :run_id AND fund_id = :fund_id
                        """),
                        {"run_id": run_id, "fund_id": fund_id}
                    )
                else:
                    result = session.execute(
                        text("""
                            SELECT * FROM backtest_fund_metrics 
                            WHERE run_id = :run_id
                        """),
                        {"run_id": run_id}
                    )
                
                return [dict(row._mapping) for row in result]
            except Exception as e:
                # Table might not exist yet
                logger.warning(f"Could not fetch fund metrics: {e}")
                return []
    
    def _ensure_metrics_table(self):
        """Create the fund metrics table if it doesn't exist."""
        with self.get_session() as session:
            session.execute(text("""
                CREATE TABLE IF NOT EXISTS backtest_fund_metrics (
                    run_id TEXT NOT NULL,
                    fund_id TEXT NOT NULL,
                    period_start TEXT,
                    period_end TEXT,
                    n_trades INTEGER DEFAULT 0,
                    n_winning_trades INTEGER DEFAULT 0,
                    n_losing_trades INTEGER DEFAULT 0,
                    avg_return REAL DEFAULT 0.0,
                    median_return REAL DEFAULT 0.0,
                    max_return REAL DEFAULT 0.0,
                    min_return REAL DEFAULT 0.0,
                    hit_rate REAL DEFAULT 0.0,
                    brier_score REAL DEFAULT 0.0,
                    turnover REAL DEFAULT 0.0,
                    avg_holding_days REAL DEFAULT 0.0,
                    avg_slippage_bps REAL DEFAULT 0.0,
                    max_drawdown REAL DEFAULT 0.0,
                    created_at TIMESTAMP,
                    PRIMARY KEY (run_id, fund_id)
                )
            """))
            session.commit()
    
    def get_statistics(self, run_id: str) -> Dict:
        """Get summary statistics for a run."""
        with self.get_session() as session:
            # Trade stats
            trade_stats = session.execute(
                text("""
                    SELECT 
                        fund_id,
                        COUNT(*) as total_trades,
                        SUM(CASE WHEN side = 'buy' THEN 1 ELSE 0 END) as buys,
                        SUM(CASE WHEN side = 'sell' THEN 1 ELSE 0 END) as sells,
                        SUM(commission) as total_commission,
                        AVG(confidence) as avg_confidence
                    FROM backtest_trades
                    WHERE run_id = :run_id
                    GROUP BY fund_id
                """),
                {"run_id": run_id}
            ).fetchall()
            
            # Decision stats
            decision_stats = session.execute(
                text("""
                    SELECT 
                        fund_id,
                        action,
                        COUNT(*) as count,
                        AVG(confidence) as avg_confidence
                    FROM backtest_decisions
                    WHERE run_id = :run_id
                    GROUP BY fund_id, action
                """),
                {"run_id": run_id}
            ).fetchall()
            
            # Final performance
            final_snapshots = session.execute(
                text("""
                    SELECT fund_id, total_value, cumulative_return, max_drawdown, sharpe_ratio
                    FROM backtest_portfolio_snapshots
                    WHERE run_id = :run_id
                    AND snapshot_date = (
                        SELECT MAX(snapshot_date) 
                        FROM backtest_portfolio_snapshots 
                        WHERE run_id = :run_id
                    )
                """),
                {"run_id": run_id}
            ).fetchall()
            
            return {
                "trade_stats": [dict(row._mapping) for row in trade_stats],
                "decision_stats": [dict(row._mapping) for row in decision_stats],
                "final_performance": [dict(row._mapping) for row in final_snapshots],
            }


# Global instance
_persistence: Optional[BacktestPersistence] = None


def get_persistence() -> BacktestPersistence:
    """Get or create the global persistence instance."""
    global _persistence
    if _persistence is None:
        _persistence = BacktestPersistence()
    return _persistence
