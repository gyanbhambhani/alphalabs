"""
Debate Runner - Goldman Sachs Quant Strategy Architect

Investment committee debate system with institutional-grade quantitative framing.
Agents act as managing directors on Goldman Sachs' algorithmic trading desk,
designing systematic strategies with full strategy memos.

Key Features:
1. DATA-FIRST: Screen universe with strategy-specific signals before debate
2. GS PERSONA: Strategy thesis, entry/exit rules, position sizing, risk params
3. COLLABORATIVE: Analyst and Critic converge on ONE stock with validated signals
4. ACCOUNTABLE: Decisions traceable to specific features and evidence

Debate Flow:
1. Screen: Rank stocks by momentum, mean_reversion, or volatility signals
2. Present: Top K candidates to both agents with full feature data
3. Debate: Multi-round discussion in Goldman Sachs strategy memo format
4. Execute: Trade the agreed stock with validated confidence scoring
"""

from datetime import date
from typing import Dict, List, Optional, Any, Set
import logging

from langchain_core.messages import SystemMessage, HumanMessage

# Use centralized logging
from app.logging_config import debate_log, signals_log, consensus_log

# Import shared utilities
from core.backtest.debate_common import (
    DebateMessage,
    TradingDecision,
    get_llm,
    extract_json,
)

# Import screening utilities
from core.backtest.screening import (
    screen_universe_for_strategy,
    ScreenedCandidate,
    get_feature_value,
    DecisionAuditTrail,
)

# Import collaboration framework
from core.collaboration.debate_v2 import (
    AVAILABLE_FEATURES,
    THESIS_REQUIRED_EVIDENCE,
    ThesisType,
    InvalidationRule,
    EvidenceReference,
    ThesisProposal,
    ConfidenceDecomposition,
    DialogMove,
    Counterfactual,
    AgentTurn,
    ConsensusBoard,
    RiskManagerDecision,
    validate_evidence,
    update_consensus_board,
    compute_conservative_weight,
)

# Import experience memory
from core.ai.experience_memory import ExperienceStore, generate_accountability_brief

from core.data.snapshot import GlobalMarketSnapshot
from core.backtest.portfolio_tracker import BacktestPortfolio

logger = logging.getLogger(__name__)


class CollaborativeDebateRunner:
    """
    Investment committee debate system - Goldman Sachs Quant Architect style.

    DATA-FIRST: Screen entire universe, rank by actual signals
    - PRESENT CANDIDATES: Both agents see the SAME top candidates
    - DEBATE ON DATA: Agents discuss specific stocks with real numbers
    - NO GUESSING: Agents pick from pre-screened list only

    Debate flow:
    1. Screen: Rank all stocks by strategy-specific signals
    2. Present: Show top 5 candidates to both agents with full data
    3. Debate: Agents discuss and converge on ONE candidate
    4. Execute: Trade the agreed stock with validated signals
    """

    MAX_ROUNDS = 3
    MIN_CONFIDENCE_COMPONENT = 0.5
    MIN_EDGE_THRESHOLD = 0.6

    def __init__(
        self,
        analyst_model: str = "gpt-4o-mini",
        critic_model: str = "claude-3-haiku-20240307",
        experience_db_path: str = "data/experience.db",
        disable_experience_store: bool = False,
    ):
        """
        Initialize collaborative debate runner.

        Args:
            analyst_model: Model for analyst agent
            critic_model: Model for critic agent
            experience_db_path: Path to experience database
            disable_experience_store: If True, disable experience retrieval
                to prevent in-sample bias during backtesting
        """
        self.analyst_model = analyst_model
        self.critic_model = critic_model
        self.disable_experience_store = disable_experience_store

        # Only create experience store if not disabled
        if disable_experience_store:
            self.experience_store = None
            logger.info(
                "ExperienceStore DISABLED for backtest mode (prevents in-sample bias)"
            )
        else:
            self.experience_store = ExperienceStore(experience_db_path)

    def _has_sufficient_data(self, snapshot: GlobalMarketSnapshot) -> bool:
        """
        Check if snapshot has sufficient data for debate.

        Requires return_21d or return_5d for at least some symbols.
        Early simulation periods won't have this data.
        """
        symbols_with_21d = 0
        symbols_with_5d = 0

        for symbol in snapshot.coverage_symbols[:20]:
            if snapshot.get_return(symbol, "21d") is not None:
                symbols_with_21d += 1
            if snapshot.get_return(symbol, "5d") is not None:
                symbols_with_5d += 1

        return symbols_with_21d >= 3 or symbols_with_5d >= 5

    async def run_debate(
        self,
        fund_id: str,
        fund_name: str,
        fund_thesis: str,
        portfolio: BacktestPortfolio,
        snapshot: GlobalMarketSnapshot,
        simulation_date: date,
        trade_budget: Optional[Any] = None,
    ) -> TradingDecision:
        """
        Run the collaborative debate for a fund.

        Args:
            fund_id: Fund identifier
            fund_name: Human-readable fund name
            fund_thesis: Fund's investment thesis
            portfolio: Current portfolio state
            snapshot: Market data snapshot
            simulation_date: Current simulation date
            trade_budget: Optional budget constraints

        Returns:
            TradingDecision with action and reasoning
        """
        transcript: List[DebateMessage] = []
        conversation: List[AgentTurn] = []

        debate_log(f"Fund {fund_id} starting debate for {simulation_date}")
        debate_log(
            f"Portfolio: ${portfolio.total_value:,.0f}, "
            f"Cash: ${portfolio.cash:,.0f} ({portfolio.cash/portfolio.total_value:.0%}), "
            f"Positions: {len(portfolio.positions)}"
        )

        try:
            if not self._has_sufficient_data(snapshot):
                debate_log(
                    f"Insufficient data for debate on {simulation_date}, "
                    f"defaulting to HOLD",
                    level=logging.WARNING
                )
                return self._create_hold_decision(
                    "Insufficient historical data for analysis (early simulation period)",
                    transcript
                )

            # PHASE 1: DATA-FIRST SCREENING
            debate_log(f"Phase 1: Screening universe for {fund_thesis[:50]}...")

            candidates = screen_universe_for_strategy(
                snapshot=snapshot,
                strategy=fund_thesis,
                portfolio=portfolio,
                top_k=5,
            )

            if not candidates:
                debate_log(
                    "No candidates found for strategy, defaulting to HOLD",
                    level=logging.WARNING
                )
                return self._create_hold_decision(
                    f"No stocks in universe match {fund_thesis[:30]} criteria today",
                    transcript
                )

            debate_log(f"Found {len(candidates)} candidates:")
            for i, c in enumerate(candidates, 1):
                debate_log(f"  #{i} {c.symbol}: {c.reasoning} (score={c.score:.3f})")

            # PHASE 2: PRE-DEBATE SETUP
            if self.experience_store and not self.disable_experience_store:
                accountability_brief = generate_accountability_brief(
                    fund_id, simulation_date, self.experience_store
                )
                similar_episodes = self.experience_store.retrieve_similar(
                    snapshot, fund_id=fund_id, k=3
                )
            else:
                accountability_brief = ""
                similar_episodes = {"similar": []}

            context = self._build_enhanced_context_with_candidates(
                fund_name, fund_thesis, portfolio, snapshot,
                simulation_date, accountability_brief, similar_episodes,
                candidates,
            )

            valid_symbols = {c.symbol for c in candidates}
            candidate_signals = {c.symbol: c.signals for c in candidates}

            # PHASE 3: COMMITTEE DEBATE
            debate_log(f"Phase 2: Committee debate on {len(candidates)} candidates...")
            consensus_board = ConsensusBoard()

            for round_num in range(self.MAX_ROUNDS):
                debate_log(
                    f"Round {round_num + 1}/{self.MAX_ROUNDS} - Analyst proposing..."
                )

                analyst_turn, analyst_errors = await self._get_agent_turn_with_repair(
                    agent_id="analyst",
                    model=self.analyst_model,
                    role="analyst",
                    round_num=round_num,
                    context=context,
                    conversation=conversation,
                    fund_thesis=fund_thesis,
                    snapshot=snapshot,
                    transcript=transcript,
                )

                if analyst_turn:
                    debate_log(
                        f"Analyst: {analyst_turn.action.upper()} "
                        f"{analyst_turn.symbol or 'N/A'} @ "
                        f"{analyst_turn.suggested_weight:.0%} weight, "
                        f"thesis={analyst_turn.thesis.thesis_type.value}, "
                        f"confidence={analyst_turn.confidence.overall():.0%}"
                    )

                debate_log(
                    f"Round {round_num + 1}/{self.MAX_ROUNDS} - Critic responding..."
                )

                critic_turn, critic_errors = await self._get_agent_turn_with_repair(
                    agent_id="critic",
                    model=self.critic_model,
                    role="critic",
                    round_num=round_num,
                    context=context,
                    conversation=conversation,
                    fund_thesis=fund_thesis,
                    snapshot=snapshot,
                    transcript=transcript,
                )

                if critic_turn:
                    debate_log(
                        f"Critic: {critic_turn.action.upper()} "
                        f"{critic_turn.symbol or 'N/A'} @ "
                        f"{critic_turn.suggested_weight:.0%} weight, "
                        f"thesis={critic_turn.thesis.thesis_type.value}, "
                        f"confidence={critic_turn.confidence.overall():.0%}"
                    )

                if analyst_turn is None or critic_turn is None:
                    debate_log(
                        f"Agent failed validation, defaulting to HOLD. "
                        f"Errors: {analyst_errors + critic_errors}",
                        level=logging.WARNING
                    )
                    return self._create_hold_decision(
                        f"Validation failed: {analyst_errors + critic_errors}",
                        transcript
                    )

                conversation.extend([analyst_turn, critic_turn])

                consensus_board = update_consensus_board(
                    consensus_board, analyst_turn, critic_turn, snapshot,
                    fund_strategy=fund_thesis,
                    valid_symbols=valid_symbols,
                )

                gate = consensus_board.gate
                consensus_log(f"Computing gate for {fund_id}...")
                consensus_log(
                    f"action_match={gate.action_match}, "
                    f"symbol_match={gate.symbol_match}, "
                    f"symbol_in_candidates={gate.symbol_in_candidates}, "
                    f"thesis_type_match={gate.thesis_type_match}"
                )
                consensus_log(
                    f"fund_strategy_match={gate.fund_strategy_match} "
                    f"({fund_thesis.split()[0]} -> "
                    f"{analyst_turn.thesis.thesis_type.value})"
                )
                min_conf = min(
                    analyst_turn.confidence.min_component(),
                    critic_turn.confidence.min_component(),
                )
                consensus_log(
                    f"min_confidence_met={gate.min_confidence_met} "
                    f"({min_conf:.2f} >= 0.5)"
                )
                min_edge = min(
                    analyst_turn.confidence.signal_strength,
                    critic_turn.confidence.signal_strength,
                )
                consensus_log(
                    f"min_edge_met={gate.min_edge_met} ({min_edge:.2f} >= 0.6)"
                )

                if consensus_board.can_execute():
                    consensus_log(
                        f"Gate PASSED - all {len(gate.passed_gates())} checks OK",
                        level=logging.INFO
                    )
                    break
                else:
                    consensus_log(
                        f"Gate FAILED - {len(gate.failed_gates())} checks failed: "
                        f"{gate.failed_gates()}",
                        level=logging.WARNING
                    )

            # 4. Finalize decision
            if consensus_board.can_execute():
                decision = self._build_consensus_decision(
                    consensus_board, analyst_turn, critic_turn,
                    portfolio, transcript, snapshot, fund_id,
                    valid_symbols, candidate_signals,
                )
                debate_log(
                    f"Final decision: {decision.action.upper()} "
                    f"{decision.symbol or 'N/A'} @ "
                    f"{decision.target_weight or 0:.0%}, "
                    f"confidence={decision.confidence:.0%}"
                )
            else:
                consensus_log(
                    f"No consensus after {self.MAX_ROUNDS} rounds",
                    level=logging.WARNING
                )
                if consensus_board.gate.rejection_reasons:
                    consensus_log("Rejection reasons:")
                    for reason in consensus_board.gate.rejection_reasons:
                        consensus_log(f"  - {reason}")
                decision = self._create_hold_decision(
                    f"No consensus: {consensus_board.gate.failed_gates()}",
                    transcript
                )

            # 5. Record experience (skip in backtest mode)
            if (analyst_turn and critic_turn and
                    self.experience_store and not self.disable_experience_store):
                self.experience_store.record(
                    snapshot=snapshot,
                    fund_id=fund_id,
                    action=decision.action,
                    symbol=decision.symbol,
                    thesis_type=analyst_turn.thesis.thesis_type,
                    confidence=decision.confidence,
                    conversation=conversation,
                )

            return decision

        except Exception as e:
            logger.error(f"[{fund_id}] Debate error: {e}")
            return self._create_hold_decision(
                f"Debate error: {str(e)[:100]}",
                transcript
            )

    async def _get_agent_turn_with_repair(
        self,
        agent_id: str,
        model: str,
        role: str,
        round_num: int,
        context: str,
        conversation: List[AgentTurn],
        fund_thesis: str,
        snapshot: GlobalMarketSnapshot,
        transcript: List[DebateMessage],
        max_repairs: int = 1,
    ) -> tuple[Optional[AgentTurn], List[str]]:
        """Get agent turn with one repair attempt on failure."""
        errors: List[str] = []
        prompt = self._build_agent_prompt(
            role, round_num, context, conversation, fund_thesis
        )

        for attempt in range(max_repairs + 1):
            try:
                llm = get_llm(model, temperature=0.5)

                response = await llm.ainvoke([
                    SystemMessage(content=self._get_agent_system_prompt(
                        role, fund_thesis
                    )),
                    HumanMessage(content=prompt),
                ])

                content = (
                    response.content
                    if hasattr(response, 'content')
                    else str(response)
                )

                transcript.append(DebateMessage(
                    phase=f"round_{round_num}_{role}",
                    model=model,
                    content=content,
                    tokens_used=0,
                ))

                data = extract_json(content)
                if not data:
                    errors.append(f"Attempt {attempt + 1}: Failed to parse JSON")
                    if attempt < max_repairs:
                        prompt = self._build_repair_prompt(errors[-1])
                    continue

                turn = self._parse_agent_turn(data, agent_id, round_num)

                valid, evidence_errors = validate_evidence(
                    turn.evidence_cited,
                    turn.thesis.thesis_type,
                    snapshot
                )

                if not valid:
                    errors.extend(evidence_errors)
                    if attempt < max_repairs:
                        prompt = self._build_evidence_repair_prompt(
                            evidence_errors, turn.thesis.thesis_type
                        )
                    continue

                return turn, []

            except Exception as e:
                errors.append(f"Attempt {attempt + 1}: {str(e)[:100]}")
                if attempt < max_repairs:
                    prompt = self._build_repair_prompt(errors[-1])

        return None, errors

    def _build_enhanced_context(
        self,
        fund_name: str,
        fund_thesis: str,
        portfolio: BacktestPortfolio,
        snapshot: GlobalMarketSnapshot,
        simulation_date: date,
        accountability_brief: str,
        similar_episodes: Dict[str, List],
    ) -> str:
        """Build enhanced context with accountability and precedents."""
        positions_str = ""
        for sym, pos in portfolio.positions.items():
            weight = portfolio.get_position_weight(sym)
            positions_str += (
                f"  - {sym}: {pos.quantity:.0f} shares @ "
                f"${pos.current_price:.2f} "
                f"({weight:.1%}, {pos.unrealized_return:+.1%})\n"
            )

        if not positions_str:
            positions_str = "  (No positions - 100% cash)\n"

        market_data = []
        available_in_snapshot = set()

        for sym in list(snapshot.prices.keys())[:20]:
            parts = [sym]
            ret_1d = snapshot.get_return(sym, "1d")
            ret_5d = snapshot.get_return(sym, "5d")
            ret_21d = snapshot.get_return(sym, "21d")
            ret_63d = snapshot.get_return(sym, "63d")
            vol_5d = snapshot.get_volatility(sym, "5d")
            vol_21d = snapshot.get_volatility(sym, "21d")

            if ret_1d is not None:
                parts.append(f"1d={ret_1d:+.1%}")
                available_in_snapshot.add("return_1d")
            if ret_5d is not None:
                parts.append(f"5d={ret_5d:+.1%}")
                available_in_snapshot.add("return_5d")
            if ret_21d is not None:
                parts.append(f"21d={ret_21d:+.1%}")
                available_in_snapshot.add("return_21d")
            if ret_63d is not None:
                parts.append(f"63d={ret_63d:+.1%}")
                available_in_snapshot.add("return_63d")
            if vol_5d is not None:
                parts.append(f"vol5d={vol_5d:.1%}")
                available_in_snapshot.add("volatility_5d")
            if vol_21d is not None:
                parts.append(f"vol21d={vol_21d:.1%}")
                available_in_snapshot.add("volatility_21d")

            if len(parts) > 1:
                market_data.append(": ".join([parts[0], ", ".join(parts[1:])]))

        market_str = "\n".join(f"  {d}" for d in market_data[:10])
        features_available = (
            sorted(available_in_snapshot) if available_in_snapshot else ["price"]
        )

        episodes_str = ""
        for ep in similar_episodes.get("similar", [])[:2]:
            episodes_str += (
                f"  - {ep.record_date}: {ep.action} {ep.symbol or ''} "
                f"({ep.thesis_type}) -> {ep.outcome_5d or 0:+.1%}\n"
            )

        if not episodes_str:
            episodes_str = "  (No similar episodes found)\n"

        return f"""DATE: {simulation_date}
FUND: {fund_name}
THESIS: {fund_thesis}

{accountability_brief}

PORTFOLIO:
  Total Value: ${portfolio.total_value:,.2f}
  Cash: ${portfolio.cash:,.2f} ({portfolio.cash/portfolio.total_value:.1%})
{positions_str}

FEATURES AVAILABLE TODAY: {', '.join(features_available)}
(Only cite features from this list - others have no data yet)

MARKET DATA:
{market_str}

SIMILAR PAST EPISODES:
{episodes_str}

AVAILABLE THESIS TYPES: {', '.join(t.value for t in ThesisType)}"""

    def _build_enhanced_context_with_candidates(
        self,
        fund_name: str,
        fund_thesis: str,
        portfolio: BacktestPortfolio,
        snapshot: GlobalMarketSnapshot,
        simulation_date: date,
        accountability_brief: str,
        similar_episodes: Dict[str, List],
        candidates: List[ScreenedCandidate],
    ) -> str:
        """Build context with PRE-SCREENED candidates."""
        positions_str = ""
        for sym, pos in portfolio.positions.items():
            weight = portfolio.get_position_weight(sym)
            positions_str += (
                f"  - {sym}: {pos.quantity:.0f} shares @ "
                f"${pos.current_price:.2f} "
                f"({weight:.1%}, {pos.unrealized_return:+.1%})\n"
            )

        if not positions_str:
            positions_str = "  (No positions - 100% cash)\n"

        candidates_str = ""
        for i, c in enumerate(candidates, 1):
            signals_str = ", ".join(
                f"{k}={v:+.1%}" if "return" in k or "volatility" in k else f"{k}=${v:.2f}"
                for k, v in c.signals.items()
                if v is not None
            )
            candidates_str += (
                f"  #{i} {c.symbol} (score={c.score:.2f})\n"
                f"      Signals: {signals_str}\n"
                f"      Fit: {c.thesis_fit} - {c.reasoning}\n"
            )

        episodes_str = ""
        for ep in similar_episodes.get("similar", [])[:2]:
            episodes_str += (
                f"  - {ep.record_date}: {ep.action} {ep.symbol or ''} "
                f"({ep.thesis_type}) -> {ep.outcome_5d or 0:+.1%}\n"
            )

        if not episodes_str:
            episodes_str = "  (No similar episodes found)\n"

        thesis_type = candidates[0].thesis_fit if candidates else "unknown"
        valid_symbols = [c.symbol for c in candidates]

        return f"""DATE: {simulation_date}
FUND: {fund_name}
THESIS: {fund_thesis}

{accountability_brief}

PORTFOLIO:
  Total Value: ${portfolio.total_value:,.2f}
  Cash: ${portfolio.cash:,.2f} ({portfolio.cash/portfolio.total_value:.1%})
{positions_str}

============================================================
PRE-SCREENED CANDIDATES (ranked by {thesis_type} signals)
============================================================
These stocks passed quantitative screening. You MUST pick from this list.
Do NOT suggest stocks outside this list - they failed screening.

{candidates_str}
VALID SYMBOLS: {', '.join(valid_symbols)}

============================================================

SIMILAR PAST EPISODES:
{episodes_str}

RULES:
1. You MUST select a symbol from the VALID SYMBOLS list above
2. If you disagree with all candidates, recommend HOLD
3. Cite the actual signal values shown above in your evidence
4. Thesis type must be: {thesis_type}

AVAILABLE THESIS TYPES: {', '.join(t.value for t in ThesisType)}"""

    def _get_agent_system_prompt(self, role: str, fund_thesis: str) -> str:
        """Get system prompt for agent role (Goldman Sachs Quant Architect)."""
        gs_persona = (
            "You are a managing director on Goldman Sachs' algorithmic trading "
            "desk who designs systematic trading strategies managing $10B+ in "
            "institutional capital across global equity markets."
        )
        if role == "analyst":
            role_desc = f"""{gs_persona}
You are the ANALYST. PROPOSE the best stock from the screened candidates.
Frame your reasoning as a quantitative strategy memo:
- Strategy thesis: the market inefficiency this pick exploits
- Universe selection: why this instrument fits the strategy
- Signal generation: exact rules producing the buy signal
- Entry rules: conditions that must ALL be true before opening
- Exit rules: profit targets, stop losses, invalidation conditions
- Position sizing: conviction-based allocation rationale
- Risk parameters: max drawdown, sector limits, correlation
Pick ONE stock with quantitative evidence. Be decisive."""
        else:
            role_desc = f"""{gs_persona}
You are the CRITIC. EVALUATE the analyst's proposal and reach CONSENSUS.
Apply institutional risk standards:
- Validate strategy thesis and signal logic
- Stress-test entry/exit rules and edge decay scenarios
- If the analyst's pick is reasonable, AGREE (use the SAME symbol)
- Only propose a different symbol if there's a CLEAR risk or logic flaw
- Default to AGREEING unless you have strong evidence against"""

        base = f"""{role_desc}

Fund thesis: {fund_thesis}

You must respond with ONLY valid JSON in this exact format:
{{
    "dialog_move": {{
        "acknowledge": "Restate colleague's key claim",
        "challenge": "One specific disagreement or null",
        "request": "Ask for missing evidence or null",
        "concede_or_hold": "What you updated vs stayed firm on"
    }},
    "action": "buy" or "sell" or "hold",
    "symbol": "TICKER" or null,
    "suggested_weight": 0.10,
    "risk_posture": "normal" or "defensive" or "aggressive",
    "thesis": {{
        "thesis_type": "momentum" or "mean_reversion" or "volatility",
        "horizon_days": 5,
        "primary_signal": "return_1d",
        "secondary_signal": "volatility_21d" or null,
        "invalidation_rules": [
            {{"feature": "return_1d", "symbol": "AAPL", "operator": ">", "value": 0.05}}
        ]
    }},
    "confidence": {{
        "signal_strength": 0.7,
        "regime_fit": 0.6,
        "risk_comfort": 0.55,
        "execution_feasibility": 0.8
    }},
    "evidence_cited": [
        {{"feature": "return_1d", "symbol": "AAPL"}},
        {{"feature": "volatility_21d", "symbol": "AAPL"}}
    ],
    "counterfactual": {{
        "alternative_action": "hold",
        "why_rejected": "Brief reason"
    }}
}}

CRITICAL RULES:
1. SYMBOL SELECTION: You MUST pick from the "VALID SYMBOLS" list in the context
   - These stocks passed quantitative screening for this strategy
   - Do NOT suggest any symbol outside that list - they failed screening
   - If you disagree with ALL candidates, recommend HOLD with symbol=null
2. Use the ACTUAL signal values shown in the PRE-SCREENED CANDIDATES section
   - Copy the exact numbers (return_1d, return_5d, etc.) from the candidate data
   - Do not make up signal values
3. Match thesis_type to the fund's strategy:
   - momentum: needs return_21d or return_63d
   - mean_reversion: needs return_1d or return_5d
   - volatility: needs volatility_5d or volatility_21d
4. All confidence components should be 0.0-1.0
5. HORIZON ALIGNMENT: Use standard horizons:
   - Short-term: 5 days (for mean_reversion)
   - Medium-term: 21 days (for momentum)
6. CONSENSUS IS THE GOAL:
   - If you're the CRITIC and the analyst's pick is reasonable, AGREE with it
   - Use the SAME symbol as the analyst unless there's a clear problem
   - Don't switch symbols just to be different"""

        if role == "analyst":
            return base + """

As ANALYST, focus on:
- Pick the BEST candidate from the pre-screened list
- Cite the specific signal values shown for that candidate
- Explain why this candidate fits the fund's thesis"""
        else:
            return base + """

As CRITIC, focus on:
- RESPOND TO THE ANALYST'S PICK - discuss the SAME symbol
- Challenge: Is this really the best candidate from the list?
- If you agree, confirm. If you disagree, explain why another candidate is better
- Do NOT pick a random different symbol - stay focused on the debate"""

    def _build_agent_prompt(
        self,
        role: str,
        round_num: int,
        context: str,
        conversation: List[AgentTurn],
        fund_thesis: str,
    ) -> str:
        """Build prompt for agent turn."""
        history = ""
        last_analyst_symbol = None
        for turn in conversation:
            history += f"\n{turn.agent_id.upper()} (Round {turn.round_num}):\n"
            history += f"  Action: {turn.action} {turn.symbol or ''}\n"
            history += f"  Thesis: {turn.thesis.thesis_type.value}\n"
            history += f"  Confidence: {turn.confidence.overall():.0%}\n"
            if turn.agent_id == "analyst" and turn.symbol:
                last_analyst_symbol = turn.symbol

        if not history:
            history = "(This is the first round - no prior conversation)"

        if role == "analyst":
            task_prompt = f"""YOUR TASK (Round {round_num + 1}):
1. If there's prior conversation, ACKNOWLEDGE your colleague's feedback
2. Pick the BEST stock from the PRE-SCREENED CANDIDATES list
3. Provide your proposal with thesis, evidence, and confidence
4. If the critic disagreed with your previous pick, either:
   - DEFEND your pick with stronger evidence, OR
   - CONCEDE and switch to their suggested alternative

Respond with JSON only."""
        else:
            if last_analyst_symbol:
                task_prompt = f"""YOUR TASK (Round {round_num + 1}):
*** CRITICAL: The analyst proposed {last_analyst_symbol}. You MUST respond about {last_analyst_symbol}. ***

1. EVALUATE the analyst's pick of {last_analyst_symbol}
2. Either AGREE with {last_analyst_symbol} (use the SAME symbol in your response)
   OR explain why you disagree and suggest ONE alternative from the candidates
3. If you AGREE: Use symbol="{last_analyst_symbol}" in your JSON response
4. If you DISAGREE: Explain specifically why, then suggest your alternative

DO NOT propose a different symbol unless you have a STRONG reason to reject {last_analyst_symbol}.
The goal is CONSENSUS - try to agree unless there's a clear problem.

Respond with JSON only."""
            else:
                task_prompt = f"""YOUR TASK (Round {round_num + 1}):
1. Review the PRE-SCREENED CANDIDATES
2. Provide your assessment and pick
3. Provide your proposal with thesis, evidence, and confidence

Respond with JSON only."""

        return f"""{context}

CONVERSATION SO FAR:
{history}

{task_prompt}"""

    def _build_repair_prompt(self, error: str) -> str:
        """Build repair prompt after validation failure."""
        return f"""Your previous response had an error:
{error}

Please fix and respond again with valid JSON.
Remember:
- evidence_cited must use ONLY: {', '.join(sorted(AVAILABLE_FEATURES))}
- You need at least 2 evidence items
- All fields are required"""

    def _build_evidence_repair_prompt(
        self,
        errors: List[str],
        thesis_type: ThesisType,
    ) -> str:
        """Build repair prompt for evidence validation failure."""
        required = THESIS_REQUIRED_EVIDENCE.get(thesis_type, set())
        return f"""Your evidence citations had errors:
{chr(10).join(errors)}

Available features: {', '.join(sorted(AVAILABLE_FEATURES))}
Required for {thesis_type.value}: {', '.join(required)}

Please fix and respond again with valid JSON."""

    def _parse_agent_turn(
        self,
        data: Dict[str, Any],
        agent_id: str,
        round_num: int,
    ) -> AgentTurn:
        """Parse agent turn from JSON data."""
        dm_data = data.get("dialog_move", {})
        dialog_move = DialogMove(
            acknowledge=dm_data.get("acknowledge", ""),
            challenge=dm_data.get("challenge"),
            request=dm_data.get("request"),
            concede_or_hold=dm_data.get("concede_or_hold", ""),
        )

        thesis_data = data.get("thesis", {})
        invalidation_rules = [
            InvalidationRule(
                feature=r.get("feature", ""),
                symbol=r.get("symbol", ""),
                operator=r.get("operator", ">"),
                value=float(r.get("value", 0)),
            )
            for r in thesis_data.get("invalidation_rules", [])
        ]

        thesis = ThesisProposal(
            thesis_type=ThesisType(thesis_data.get("thesis_type", "momentum")),
            horizon_days=int(thesis_data.get("horizon_days", 5)),
            primary_signal=thesis_data.get("primary_signal", "return_1d"),
            secondary_signal=thesis_data.get("secondary_signal"),
            invalidation_rules=invalidation_rules,
        )

        conf_data = data.get("confidence", {})
        confidence = ConfidenceDecomposition(
            signal_strength=float(conf_data.get("signal_strength", 0.5)),
            regime_fit=float(conf_data.get("regime_fit", 0.5)),
            risk_comfort=float(conf_data.get("risk_comfort", 0.5)),
            execution_feasibility=float(conf_data.get("execution_feasibility", 0.5)),
        )

        evidence = [
            EvidenceReference(
                feature=e.get("feature", ""),
                symbol=e.get("symbol", ""),
            )
            for e in data.get("evidence_cited", [])
        ]

        cf_data = data.get("counterfactual", {})
        counterfactual = Counterfactual(
            alternative_action=cf_data.get("alternative_action", "hold"),
            why_rejected=cf_data.get("why_rejected", ""),
        )

        return AgentTurn(
            agent_id=agent_id,
            round_num=round_num,
            dialog_move=dialog_move,
            action=data.get("action", "hold"),
            symbol=data.get("symbol"),
            suggested_weight=float(data.get("suggested_weight", 0.0)),
            risk_posture=data.get("risk_posture", "normal"),
            thesis=thesis,
            confidence=confidence,
            evidence_cited=evidence,
            counterfactual=counterfactual,
        )

    def _build_consensus_decision(
        self,
        consensus_board: ConsensusBoard,
        analyst_turn: AgentTurn,
        critic_turn: AgentTurn,
        portfolio: BacktestPortfolio,
        transcript: List[DebateMessage],
        snapshot: GlobalMarketSnapshot,
        fund_id: str,
        valid_symbols: Set[str],
        candidate_signals: Dict[str, Dict[str, float]],
    ) -> TradingDecision:
        """Build final decision from consensus."""
        action = consensus_board.agreed_action or "hold"
        symbol = consensus_board.agreed_symbol

        if action != "hold" and symbol:
            if symbol not in valid_symbols:
                debate_log(
                    f"BLOCKING trade - {symbol} not in screened candidates: "
                    f"{valid_symbols}",
                    level=logging.WARNING
                )
                return self._create_hold_decision(
                    f"Trade blocked: {symbol} not in pre-screened candidates",
                    transcript
                )

            if symbol in candidate_signals:
                signals_log(
                    f"Using pre-screened signals for {symbol}: "
                    f"{candidate_signals[symbol]}"
                )

        thesis_type = (
            consensus_board.agreed_thesis_type or
            analyst_turn.thesis.thesis_type
        )

        if symbol and symbol in candidate_signals:
            screened_signals = candidate_signals[symbol]
            audit = DecisionAuditTrail(
                signals_used={
                    f"{symbol}_{k}": v
                    for k, v in screened_signals.items()
                    if v is not None
                },
                evidence_used=[
                    {"symbol": symbol, "feature": k, "value": v, "source": "screening"}
                    for k, v in screened_signals.items()
                    if v is not None
                ],
                validation_report={
                    "passed": True,
                    "errors": [],
                    "missing_required": [],
                    "source": "pre_screened_candidates",
                },
            )
            signals_log(
                f"Using {len(audit.signals_used)} pre-screened signals for {symbol}"
            )
        else:
            audit = self._extract_signals_from_evidence(
                analyst_turn, critic_turn, snapshot, thesis_type
            )

        if action != "hold":
            if not audit.signals_used:
                debate_log(
                    "BLOCKING trade - no validated signals attached",
                    level=logging.WARNING
                )
                return self._create_hold_decision(
                    "Trade blocked: no validated signals attached",
                    transcript
                )

            if not audit.validation_report["passed"]:
                debate_log(
                    f"BLOCKING trade - validation failed: "
                    f"{audit.validation_report['errors']}",
                    level=logging.WARNING
                )
                return self._create_hold_decision(
                    f"Trade blocked: {audit.validation_report['errors']}",
                    transcript
                )

        suggested_weight = compute_conservative_weight(analyst_turn, critic_turn)

        risk_decision = RiskManagerDecision.compute(
            action=action,
            symbol=consensus_board.agreed_symbol,
            suggested_weight=suggested_weight,
            risk_posture=analyst_turn.risk_posture,
            portfolio_cash=portfolio.cash,
            portfolio_total_value=portfolio.total_value,
            current_position_weight=portfolio.get_position_weight(
                consensus_board.agreed_symbol
            ) or 0.0,
        )

        if not risk_decision.approved:
            return self._create_hold_decision(
                f"Risk manager rejected: {risk_decision.constraints_hit}",
                transcript
            )

        reasoning = (
            f"Consensus: {action} "
            f"{consensus_board.agreed_symbol or ''} "
            f"({thesis_type.value if thesis_type else 'unknown'}) "
            f"| Weight: {risk_decision.final_weight:.1%}"
        )

        if risk_decision.constraints_hit:
            reasoning += f" | Constraints: {', '.join(risk_decision.constraints_hit)}"

        return TradingDecision(
            action=action,
            symbol=consensus_board.agreed_symbol,
            target_weight=risk_decision.final_weight,
            reasoning=reasoning,
            confidence=min(
                analyst_turn.confidence.overall(),
                critic_turn.confidence.overall()
            ),
            signals_used=audit.signals_used,
            debate_transcript=transcript,
            models_used={
                "analyst": self.analyst_model,
                "critic": self.critic_model,
            },
        )

    def _extract_signals_from_evidence(
        self,
        analyst_turn: AgentTurn,
        critic_turn: AgentTurn,
        snapshot: GlobalMarketSnapshot,
        thesis_type: ThesisType,
    ) -> DecisionAuditTrail:
        """Extract actual feature values with thesis-specific validation."""
        signals: Dict[str, float] = {}
        evidence: List[Dict[str, Any]] = []
        errors: List[str] = []

        symbol = analyst_turn.symbol or critic_turn.symbol or "?"
        signals_log(f"Extracting signals for {symbol}...")

        for agent_id, turn in [("analyst", analyst_turn), ("critic", critic_turn)]:
            for ref in turn.evidence_cited:
                key = f"{ref.symbol}_{ref.feature}"
                value = get_feature_value(snapshot, ref.symbol, ref.feature)

                evidence.append({
                    "symbol": ref.symbol,
                    "feature": ref.feature,
                    "agent": agent_id,
                    "value": value,
                })

                if value is None:
                    errors.append(f"{agent_id} cited {key} but value is None")
                elif key not in signals:
                    signals[key] = value

        if signals:
            signal_strs = [f"{k}={v:.4f}" for k, v in list(signals.items())[:5]]
            signals_log(f"Found: {', '.join(signal_strs)}")
        else:
            signals_log("No signals found!", level=logging.WARNING)

        required = THESIS_REQUIRED_EVIDENCE.get(thesis_type, set())
        cited_features = {
            ref.feature for ref in
            analyst_turn.evidence_cited + critic_turn.evidence_cited
        }
        missing_required = required - cited_features

        signals_log(f"Thesis {thesis_type.value} requires: {required}")

        if missing_required:
            errors.append(
                f"Thesis {thesis_type.value} requires {required}, "
                f"missing: {missing_required}"
            )

        passed = len(errors) == 0 and len(signals) >= len(required)

        if passed:
            signals_log(
                f"Validation PASSED - {len(signals)} signals, "
                f"{len(missing_required)} missing"
            )
        else:
            signals_log(
                f"Validation FAILED - {len(signals)} signals, "
                f"errors: {errors}",
                level=logging.WARNING
            )

        return DecisionAuditTrail(
            signals_used=signals,
            evidence_used=evidence,
            validation_report={
                "passed": passed,
                "errors": errors,
                "signal_count": len(signals),
                "evidence_count": len(evidence),
                "required_features": list(required),
                "missing_required": list(missing_required),
            }
        )

    def _create_hold_decision(
        self,
        reason: str,
        transcript: List[DebateMessage],
    ) -> TradingDecision:
        """Create a HOLD decision."""
        return TradingDecision(
            action="hold",
            reasoning=reason,
            confidence=0.0,
            debate_transcript=transcript,
            models_used={
                "analyst": self.analyst_model,
                "critic": self.critic_model,
            },
        )


def get_debate_runner(**kwargs) -> CollaborativeDebateRunner:
    """
    Factory function for the debate runner.

    Args:
        **kwargs: Arguments for CollaborativeDebateRunner (analyst_model,
            critic_model, experience_db_path, disable_experience_store)

    Returns:
        CollaborativeDebateRunner instance
    """
    return CollaborativeDebateRunner(**kwargs)


__all__ = [
    "CollaborativeDebateRunner",
    "get_debate_runner",
    "DebateMessage",
    "TradingDecision",
    "get_llm",
    "extract_json",
]
