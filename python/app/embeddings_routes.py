"""Embeddings API endpoints"""
from fastapi import APIRouter, HTTPException, Query

from core.semantic.vector_db import VectorDatabase
from app.query_parser import parse_query
from app.schemas import (
    EmbeddingsStatsResponse, EmbeddingsListResponse, EmbeddingResponse,
    EmbeddingMetadata, EmbeddingSearchQuery, EmbeddingSearchResult
)

router = APIRouter(prefix="/api/embeddings", tags=["embeddings"])


@router.get("/stats", response_model=EmbeddingsStatsResponse)
async def get_embeddings_stats():
    """Get summary statistics about the embeddings database"""
    try:
        db = VectorDatabase(persist_directory="./chroma_data")

        total_count = db.get_count()

        if total_count == 0:
            return EmbeddingsStatsResponse(
                totalCount=0,
                dateRange=("", ""),
                avgReturn1m=0.0,
                avgVolatility21d=0.0
            )

        # Get date range
        date_range = db.get_date_range()

        # Get all data to calculate averages
        all_data = db.collection.get(include=['metadatas'])

        # Calculate averages
        returns_1m = [
            m.get('return_1m', 0)
            for m in all_data['metadatas']
        ]
        vols_21d = [
            m.get('volatility_21d', 0)
            for m in all_data['metadatas']
        ]

        avg_return = sum(returns_1m) / len(returns_1m) if returns_1m else 0
        avg_vol = sum(vols_21d) / len(vols_21d) if vols_21d else 0

        return EmbeddingsStatsResponse(
            totalCount=total_count,
            dateRange=date_range,
            avgReturn1m=avg_return,
            avgVolatility21d=avg_vol
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get stats: {str(e)}"
        )


@router.get("", response_model=EmbeddingsListResponse)
async def get_embeddings(
    page: int = Query(1, ge=1),
    per_page: int = Query(50, ge=1, le=500),
    sort_by: str = Query("date",
                         pattern="^(date|return_1m|volatility_21d|price)$"),
    order: str = Query("desc", pattern="^(asc|desc)$")
):
    """List all embeddings with pagination and sorting"""
    try:
        db = VectorDatabase(persist_directory="./chroma_data")

        # Get all data
        all_data = db.collection.get(include=['metadatas'])

        if not all_data['ids']:
            return EmbeddingsListResponse(
                embeddings=[],
                total=0,
                page=page,
                perPage=per_page
            )

        # Create list of embeddings with metadata
        embeddings_list = []
        for i, doc_id in enumerate(all_data['ids']):
            metadata = all_data['metadatas'][i]

            embeddings_list.append({
                'id': doc_id,
                'metadata': metadata,
                'sort_key': metadata.get(sort_by, doc_id)
            })

        # Sort
        reverse = (order == "desc")
        embeddings_list.sort(key=lambda x: x['sort_key'], reverse=reverse)

        # Paginate
        total = len(embeddings_list)
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        page_data = embeddings_list[start_idx:end_idx]

        # Convert to response format
        embeddings = [
            EmbeddingResponse(
                id=item['id'],
                metadata=EmbeddingMetadata(**item['metadata'])
            )
            for item in page_data
        ]

        return EmbeddingsListResponse(
            embeddings=embeddings,
            total=total,
            page=page,
            perPage=per_page
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get embeddings: {str(e)}"
        )


@router.post("/search")
async def search_embeddings(query: EmbeddingSearchQuery):
    """Natural language semantic search of market embeddings"""
    try:
        db = VectorDatabase(persist_directory="./chroma_data")

        # Parse the natural language query
        where_filter, interpretation = parse_query(query.query)

        # Get matching results
        if where_filter:
            # Query with filters
            results = db.collection.get(
                where=where_filter,
                limit=query.top_k,
                include=['metadatas']
            )
        else:
            # Get all results (no specific filters)
            all_data = db.collection.get(include=['metadatas'])

            # Limit to top_k
            results = {
                'ids': all_data['ids'][:query.top_k],
                'metadatas': all_data['metadatas'][:query.top_k]
            }

        # Convert to response format
        search_results = []
        for i, doc_id in enumerate(results['ids']):
            metadata = results['metadatas'][i]

            search_results.append(
                EmbeddingSearchResult(
                    id=doc_id,
                    metadata=EmbeddingMetadata(**metadata),
                    similarity=1.0,  # Metadata search, all matches equally
                    queryInterpretation=interpretation
                )
            )

        return {
            "results": search_results,
            "interpretation": interpretation,
            "total": len(search_results)
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {str(e)}"
        )


@router.get(
    "/stats/{symbol}",
    response_model=EmbeddingsStatsResponse
)
async def get_embeddings_stats_for_symbol(symbol: str):
    """Get embedding statistics for a specific stock"""
    try:
        db = VectorDatabase(
            persist_directory="./chroma_data",
            symbol=symbol.upper()
        )

        total_count = db.get_count()

        if total_count == 0:
            return EmbeddingsStatsResponse(
                totalCount=0,
                dateRange=("", ""),
                avgReturn1m=0.0,
                avgVolatility21d=0.0
            )

        date_range = db.get_date_range()

        # Get all data to calculate averages
        all_data = db.collection.get(include=['metadatas'])

        returns_1m = [m.get('return_1m', 0) for m in all_data['metadatas']]
        vols_21d = [
            m.get('volatility_21d', 0)
            for m in all_data['metadatas']
        ]

        avg_return = sum(returns_1m) / len(returns_1m) if returns_1m else 0
        avg_vol = sum(vols_21d) / len(vols_21d) if vols_21d else 0

        return EmbeddingsStatsResponse(
            totalCount=total_count,
            dateRange=date_range,
            avgReturn1m=avg_return,
            avgVolatility21d=avg_vol
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get stats: {str(e)}"
        )


@router.get(
    "/{symbol}",
    response_model=EmbeddingsListResponse
)
async def get_embeddings_for_symbol(
    symbol: str,
    page: int = Query(1, ge=1),
    per_page: int = Query(50, ge=1, le=500),
    sort_by: str = Query("date",
                         pattern="^(date|return_1m|volatility_21d|price)$"),
    order: str = Query("desc", pattern="^(asc|desc)$")
):
    """List embeddings for a specific stock"""
    try:
        db = VectorDatabase(
            persist_directory="./chroma_data",
            symbol=symbol.upper()
        )

        # Get all data
        all_data = db.collection.get(include=['metadatas'])

        if not all_data['ids']:
            return EmbeddingsListResponse(
                embeddings=[],
                total=0,
                page=page,
                perPage=per_page
            )

        # Create list with sort keys
        embeddings_list = []
        for i, doc_id in enumerate(all_data['ids']):
            metadata = all_data['metadatas'][i]

            embeddings_list.append({
                'id': doc_id,
                'metadata': metadata,
                'sort_key': metadata.get(sort_by, doc_id)
            })

        # Sort
        reverse = (order == "desc")
        embeddings_list.sort(key=lambda x: x['sort_key'], reverse=reverse)

        # Paginate
        total = len(embeddings_list)
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        page_data = embeddings_list[start_idx:end_idx]

        # Convert to response
        embeddings = [
            EmbeddingResponse(
                id=item['id'],
                metadata=EmbeddingMetadata(**item['metadata'])
            )
            for item in page_data
        ]

        return EmbeddingsListResponse(
            embeddings=embeddings,
            total=total,
            page=page,
            perPage=per_page
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get embeddings: {str(e)}"
        )


@router.post("/search/{symbol}")
async def search_embeddings_for_symbol(
    symbol: str,
    query: EmbeddingSearchQuery
):
    """Search embeddings for a specific stock"""
    try:
        db = VectorDatabase(
            persist_directory="./chroma_data",
            symbol=symbol.upper()
        )

        # Parse query
        where_filter, interpretation = parse_query(query.query)

        # Get matching results
        if where_filter:
            results = db.collection.get(
                where=where_filter,
                limit=query.top_k,
                include=['metadatas']
            )
        else:
            all_data = db.collection.get(include=['metadatas'])
            results = {
                'ids': all_data['ids'][:query.top_k],
                'metadatas': all_data['metadatas'][:query.top_k]
            }

        # Convert to response
        search_results = []
        for i, doc_id in enumerate(results['ids']):
            metadata = results['metadatas'][i]

            search_results.append(
                EmbeddingSearchResult(
                    id=doc_id,
                    metadata=EmbeddingMetadata(**metadata),
                    similarity=1.0,
                    queryInterpretation=interpretation
                )
            )

        return {
            "results": search_results,
            "interpretation": interpretation,
            "total": len(search_results)
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {str(e)}"
        )
