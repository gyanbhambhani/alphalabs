"""
Vector Database Interface

Uses ChromaDB for storing and searching market state embeddings.
"""
import chromadb
from chromadb.config import Settings
import numpy as np
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from pathlib import Path

from core.semantic.encoder import MarketState


@dataclass
class SearchResult:
    """Result from a semantic search"""
    date: str
    similarity: float
    metadata: Dict[str, Any]
    distance: float


class VectorDatabase:
    """
    ChromaDB-backed vector database for market state embeddings.
    
    Supports:
    - Adding market states with metadata
    - Semantic search for similar periods
    - Batch operations for efficiency
    - Multiple collections (one per stock symbol)
    """
    
    COLLECTION_NAME = "market_states"
    
    def __init__(
        self,
        persist_directory: str = "./chroma_data",
        in_memory: bool = False,
        symbol: Optional[str] = None
    ):
        """
        Initialize the vector database.
        
        Args:
            persist_directory: Where to store persistent data
            in_memory: If True, use in-memory storage (for testing)
            symbol: Stock symbol for this collection (None = default "market_states")
        """
        if in_memory:
            self.client = chromadb.Client()
        else:
            Path(persist_directory).mkdir(parents=True, exist_ok=True)
            self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Determine collection name
        self.symbol = symbol
        if symbol:
            collection_name = f"market_states_{symbol.upper()}"
        else:
            collection_name = self.COLLECTION_NAME
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )
    
    def add(self, state: MarketState) -> None:
        """
        Add a single market state to the database.
        
        Args:
            state: MarketState to add
        """
        self.collection.add(
            ids=[state.date],
            embeddings=[state.vector.tolist()],
            metadatas=[state.metadata]
        )
    
    def add_batch(self, states: List[MarketState], batch_size: int = 1000) -> int:
        """
        Add multiple market states efficiently.
        
        Args:
            states: List of MarketState objects
            batch_size: Number of states to add per batch
        
        Returns:
            Number of states added
        """
        total_added = 0
        
        for i in range(0, len(states), batch_size):
            batch = states[i:i + batch_size]
            
            self.collection.add(
                ids=[s.date for s in batch],
                embeddings=[s.vector.tolist() for s in batch],
                metadatas=[s.metadata for s in batch]
            )
            
            total_added += len(batch)
        
        return total_added
    
    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 50,
        where: Optional[Dict] = None
    ) -> List[SearchResult]:
        """
        Search for similar market states.
        
        Args:
            query_vector: Query embedding
            top_k: Number of results to return
            where: Optional filter on metadata
        
        Returns:
            List of SearchResult ordered by similarity
        """
        results = self.collection.query(
            query_embeddings=[query_vector.tolist()],
            n_results=top_k,
            where=where,
            include=["embeddings", "metadatas", "distances"]
        )
        
        search_results = []
        
        if results and results['ids'] and results['ids'][0]:
            for i, doc_id in enumerate(results['ids'][0]):
                distance = results['distances'][0][i]
                similarity = 1 - distance  # Convert distance to similarity
                
                search_results.append(SearchResult(
                    date=doc_id,
                    similarity=similarity,
                    metadata=results['metadatas'][0][i],
                    distance=distance
                ))
        
        return search_results
    
    def search_by_state(
        self,
        state: MarketState,
        top_k: int = 50,
        exclude_recent_days: int = 0
    ) -> List[SearchResult]:
        """
        Search for states similar to a given MarketState.
        
        Args:
            state: MarketState to find similar periods for
            top_k: Number of results
            exclude_recent_days: Exclude states within this many days
        
        Returns:
            List of SearchResult
        """
        return self.search(state.vector, top_k=top_k)
    
    def get_count(self) -> int:
        """Get total number of states in database"""
        return self.collection.count()
    
    def delete_all(self) -> None:
        """Delete all states (use with caution)"""
        # Get all IDs and delete
        all_data = self.collection.get()
        if all_data['ids']:
            self.collection.delete(ids=all_data['ids'])
    
    def get_date_range(self) -> tuple[str, str]:
        """Get the date range of stored states"""
        all_data = self.collection.get()
        if not all_data['ids']:
            return ("", "")
        
        dates = sorted(all_data['ids'])
        return (dates[0], dates[-1])
    
    @classmethod
    def list_all_collections(
        cls,
        persist_directory: str = "./chroma_data"
    ) -> List[str]:
        """
        List all collections in the database.
        
        Args:
            persist_directory: Where data is stored
            
        Returns:
            List of collection names
        """
        client = chromadb.PersistentClient(path=persist_directory)
        collections = client.list_collections()
        return [col.name for col in collections]
    
    @classmethod
    def get_all_symbols(
        cls,
        persist_directory: str = "./chroma_data"
    ) -> List[str]:
        """
        Get list of all stock symbols with embeddings.
        
        Args:
            persist_directory: Where data is stored
            
        Returns:
            List of stock symbols
        """
        collections = cls.list_all_collections(persist_directory)
        symbols = []
        
        for col_name in collections:
            if col_name.startswith("market_states_"):
                symbol = col_name.replace("market_states_", "")
                symbols.append(symbol)
        
        return sorted(symbols)
    
    @classmethod
    def delete_collection(
        cls,
        symbol: str,
        persist_directory: str = "./chroma_data"
    ) -> bool:
        """
        Delete a specific stock's collection.
        
        Args:
            symbol: Stock symbol
            persist_directory: Where data is stored
            
        Returns:
            True if deleted, False if not found
        """
        try:
            client = chromadb.PersistentClient(path=persist_directory)
            collection_name = f"market_states_{symbol.upper()}"
            client.delete_collection(name=collection_name)
            return True
        except Exception:
            return False
