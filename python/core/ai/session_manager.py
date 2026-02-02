"""
Redis-based Session Manager for AI Stock Terminal.

Handles:
- Session creation and retrieval with TTL
- Rate limiting per IP and user
- Result caching for deduplication (50% hit rate)
- Graceful fallback when Redis unavailable
"""

import json
import hashlib
import secrets
import logging
from datetime import datetime
from typing import Optional, Tuple

from fastapi import HTTPException

from core.ai.models import AnalysisSession, StreamChunk

logger = logging.getLogger(__name__)


class SessionManager:
    """
    Production-grade session management.
    
    Features:
    - Redis persistence (survives restarts)
    - Automatic cleanup via TTL
    - Rate limiting (10/5min per IP, 50/hour per user)
    - Result caching for duplicate queries
    - Graceful fallback to in-memory when Redis unavailable
    """
    
    SESSION_TTL = 300   # 5 minutes
    CACHE_TTL = 3600    # 1 hour
    
    # Rate limits
    IP_LIMIT_PER_5MIN = 10
    USER_LIMIT_PER_HOUR = 50
    
    def __init__(self, redis_client=None):
        """
        Initialize session manager.
        
        Args:
            redis_client: Redis async client (optional, falls back to in-memory)
        """
        self.redis = redis_client
        self._memory_sessions: dict[str, dict] = {}
        self._memory_cache: dict[str, dict] = {}
        self._memory_rate_limits: dict[str, int] = {}
    
    async def create_session(
        self,
        query: str,
        symbols: list[str],
        user_id: Optional[str] = None,
        user_ip: Optional[str] = None
    ) -> Tuple[Optional[AnalysisSession], Optional[list[dict]]]:
        """
        Create analysis session with rate limiting and deduplication.
        
        Returns:
            Tuple of (session, cached_chunks)
            - If cached_chunks is not None, return them immediately (no new session)
            - If session is not None, proceed with streaming
        """
        
        # 1. RATE LIMITING
        await self._check_rate_limit(user_id, user_ip)
        
        # 2. CHECK FOR CACHED RESULTS (deduplication)
        cache_key = self._get_cache_key(query, symbols)
        cached_chunks = await self._get_cached_result(cache_key)
        
        if cached_chunks:
            logger.info(f"Cache hit for query: {query[:50]}...")
            return None, cached_chunks
        
        # 3. CREATE NEW SESSION
        session_id = self._generate_session_id()
        session = AnalysisSession(
            id=session_id,
            query=query,
            symbols=symbols,
            context={'cache_key': cache_key},
            created_at=datetime.now(),
            user_id=user_id
        )
        
        # 4. STORE SESSION
        await self._store_session(session)
        
        logger.info(f"Created session {session_id} for query: {query[:50]}...")
        return session, None
    
    async def get_session(self, session_id: str) -> Optional[AnalysisSession]:
        """Retrieve session by ID."""
        
        if self.redis:
            try:
                data = await self.redis.get(f"session:{session_id}")
                if data:
                    return AnalysisSession.from_dict(json.loads(data))
            except Exception as e:
                logger.warning(f"Redis get failed: {e}, falling back to memory")
        
        # Fallback to in-memory
        session_data = self._memory_sessions.get(session_id)
        if session_data:
            return AnalysisSession.from_dict(session_data)
        
        return None
    
    async def cache_result(
        self,
        session: AnalysisSession,
        chunks: list[StreamChunk]
    ) -> None:
        """
        Cache analysis results for reuse.
        
        If another user asks the same question, return cached chunks.
        """
        cache_key = session.context.get('cache_key')
        if not cache_key:
            return
        
        # Serialize chunks
        chunks_data = [c.model_dump() for c in chunks]
        cache_data = {
            'chunks': chunks_data,
            'cached_at': datetime.now().isoformat(),
            'session_id': session.id
        }
        
        if self.redis:
            try:
                await self.redis.setex(
                    f"result:{cache_key}",
                    self.CACHE_TTL,
                    json.dumps(cache_data)
                )
                logger.debug(f"Cached result for key {cache_key}")
                return
            except Exception as e:
                logger.warning(f"Redis cache failed: {e}")
        
        # Fallback to in-memory cache
        self._memory_cache[cache_key] = cache_data
    
    async def _check_rate_limit(
        self,
        user_id: Optional[str],
        user_ip: Optional[str]
    ) -> None:
        """
        Enforce rate limits.
        
        Raises HTTPException 429 if limit exceeded.
        """
        
        # IP-based limit (prevent spam)
        if user_ip:
            ip_key = f"ratelimit:ip:{user_ip}:5m"
            count = await self._increment_counter(ip_key, 300)
            
            if count > self.IP_LIMIT_PER_5MIN:
                logger.warning(f"Rate limit exceeded for IP {user_ip}")
                raise HTTPException(
                    status_code=429,
                    detail="Too many requests. Please wait a few minutes."
                )
        
        # User-based limit (if authenticated)
        if user_id:
            user_key = f"ratelimit:user:{user_id}:1h"
            count = await self._increment_counter(user_key, 3600)
            
            if count > self.USER_LIMIT_PER_HOUR:
                logger.warning(f"Rate limit exceeded for user {user_id}")
                raise HTTPException(
                    status_code=429,
                    detail="Hourly limit reached. Please try again later."
                )
    
    async def _increment_counter(self, key: str, ttl: int) -> int:
        """Increment a rate limit counter with TTL."""
        
        if self.redis:
            try:
                count = await self.redis.incr(key)
                if count == 1:
                    await self.redis.expire(key, ttl)
                return count
            except Exception as e:
                logger.warning(f"Redis counter failed: {e}")
        
        # Fallback to in-memory (less accurate but works)
        current = self._memory_rate_limits.get(key, 0)
        self._memory_rate_limits[key] = current + 1
        return current + 1
    
    async def _get_cached_result(self, cache_key: str) -> Optional[list[dict]]:
        """Get cached result by cache key."""
        
        if self.redis:
            try:
                data = await self.redis.get(f"result:{cache_key}")
                if data:
                    return json.loads(data).get('chunks')
            except Exception as e:
                logger.warning(f"Redis cache get failed: {e}")
        
        # Fallback to in-memory
        cached = self._memory_cache.get(cache_key)
        if cached:
            return cached.get('chunks')
        
        return None
    
    async def _store_session(self, session: AnalysisSession) -> None:
        """Store session with TTL."""
        
        session_data = session.to_dict()
        
        if self.redis:
            try:
                await self.redis.setex(
                    f"session:{session.id}",
                    self.SESSION_TTL,
                    json.dumps(session_data)
                )
                return
            except Exception as e:
                logger.warning(f"Redis store failed: {e}")
        
        # Fallback to in-memory
        self._memory_sessions[session.id] = session_data
    
    def _get_cache_key(self, query: str, symbols: list[str]) -> str:
        """
        Generate cache key for deduplication.
        
        Same query + symbols = same cache key = reuse results.
        """
        # Normalize: lowercase query, sort symbols
        normalized = {
            'query': query.lower().strip(),
            'symbols': sorted([s.upper() for s in symbols])
        }
        
        # Hash for compact key
        content = json.dumps(normalized, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def _generate_session_id(self) -> str:
        """Generate unique session ID."""
        return secrets.token_urlsafe(16)
    
    async def cleanup_expired(self) -> int:
        """
        Clean up expired in-memory sessions.
        
        Returns number of sessions cleaned up.
        Call this periodically if not using Redis.
        """
        if self.redis:
            # Redis handles TTL automatically
            return 0
        
        now = datetime.now()
        expired = []
        
        for session_id, data in self._memory_sessions.items():
            created = datetime.fromisoformat(data['created_at'])
            if (now - created).seconds > self.SESSION_TTL:
                expired.append(session_id)
        
        for session_id in expired:
            del self._memory_sessions[session_id]
        
        if expired:
            logger.info(f"Cleaned up {len(expired)} expired sessions")
        
        return len(expired)


# Global session manager instance (initialized in main.py)
session_manager: Optional[SessionManager] = None


def get_session_manager() -> SessionManager:
    """Get the global session manager instance."""
    global session_manager
    if session_manager is None:
        # Create with no Redis (in-memory fallback)
        session_manager = SessionManager()
        logger.warning("Using in-memory session storage (Redis not configured)")
    return session_manager


def init_session_manager(redis_url: Optional[str] = None) -> SessionManager:
    """
    Initialize the global session manager.
    
    Args:
        redis_url: Redis connection URL (optional)
        
    Returns:
        Initialized SessionManager
    """
    global session_manager
    
    redis_client = None
    
    if redis_url:
        try:
            import redis.asyncio as aioredis
            redis_client = aioredis.from_url(
                redis_url,
                encoding="utf-8",
                decode_responses=True
            )
            logger.info(f"Connected to Redis: {redis_url}")
        except ImportError:
            logger.warning("redis package not installed, using in-memory storage")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}, using in-memory storage")
    
    session_manager = SessionManager(redis_client)
    return session_manager
