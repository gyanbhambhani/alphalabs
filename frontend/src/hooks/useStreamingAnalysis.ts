/**
 * Hook for streaming stock analysis from the AI Stock Terminal.
 * 
 * Two-phase approach:
 * 1. POST /api/search/session - Create session (may return cached results)
 * 2. GET /api/search/analyze-stream - SSE stream (if not cached)
 */

import { useState, useCallback, useRef, useEffect } from 'react';
import { api } from '@/lib/api';
import type { StreamChunk } from '@/types';

export type AnalysisStatus = 
  | 'idle'
  | 'creating_session'
  | 'streaming'
  | 'complete'
  | 'error';

export interface UseStreamingAnalysisReturn {
  /** Current status of the analysis */
  status: AnalysisStatus;
  
  /** All chunks received so far */
  chunks: StreamChunk[];
  
  /** Error message if status is 'error' */
  error: string | null;
  
  /** Start analysis for given query and symbols */
  analyze: (query: string, symbols: string[]) => Promise<void>;
  
  /** Cancel current analysis */
  cancel: () => void;
  
  /** Clear all chunks and reset state */
  reset: () => void;
}

export function useStreamingAnalysis(): UseStreamingAnalysisReturn {
  const [status, setStatus] = useState<AnalysisStatus>('idle');
  const [chunks, setChunks] = useState<StreamChunk[]>([]);
  const [error, setError] = useState<string | null>(null);
  
  // Keep track of active EventSource for cleanup
  const eventSourceRef = useRef<EventSource | null>(null);
  
  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
        eventSourceRef.current = null;
      }
    };
  }, []);
  
  const cancel = useCallback(() => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
      eventSourceRef.current = null;
    }
    setStatus('idle');
  }, []);
  
  const reset = useCallback(() => {
    cancel();
    setChunks([]);
    setError(null);
    setStatus('idle');
  }, [cancel]);
  
  const analyze = useCallback(async (query: string, symbols: string[]) => {
    // Reset previous state
    reset();
    setStatus('creating_session');
    
    try {
      // Phase 1: Create session
      const session = await api.createSearchSession(query, symbols);
      
      // Check if we got cached results
      if (session.cached && session.cachedChunks) {
        setChunks(session.cachedChunks);
        setStatus('complete');
        return;
      }
      
      // Phase 2: Start streaming
      if (!session.sessionId) {
        throw new Error('No session ID returned');
      }
      
      setStatus('streaming');
      
      const eventSource = api.createAnalysisStream(session.sessionId);
      eventSourceRef.current = eventSource;
      
      eventSource.onmessage = (event) => {
        try {
          const chunk: StreamChunk = JSON.parse(event.data);
          
          setChunks(prev => [...prev, chunk]);
          
          // Check for completion or error
          if (chunk.type === 'complete') {
            setStatus('complete');
            eventSource.close();
            eventSourceRef.current = null;
          } else if (chunk.type === 'error') {
            const errorContent = chunk.content as { message: string };
            setError(errorContent.message || 'Analysis failed');
            setStatus('error');
            eventSource.close();
            eventSourceRef.current = null;
          }
        } catch (parseError) {
          console.error('Failed to parse SSE chunk:', parseError);
        }
      };
      
      eventSource.onerror = (err) => {
        console.error('SSE error:', err);
        
        // Only set error if not already complete
        if (status !== 'complete') {
          setError('Connection lost. Please try again.');
          setStatus('error');
        }
        
        eventSource.close();
        eventSourceRef.current = null;
      };
      
    } catch (err) {
      console.error('Analysis failed:', err);
      setError(err instanceof Error ? err.message : 'Analysis failed');
      setStatus('error');
    }
  }, [reset, status]);
  
  return {
    status,
    chunks,
    error,
    analyze,
    cancel,
    reset,
  };
}

/**
 * Extract chunks by type for easier rendering.
 */
export function useChunksByType(chunks: StreamChunk[]) {
  const textChunks = chunks.filter(c => c.type === 'text');
  const chartChunks = chunks.filter(c => c.type === 'chart');
  const tableChunks = chunks.filter(c => c.type === 'table');
  const hasError = chunks.some(c => c.type === 'error');
  const isComplete = chunks.some(c => c.type === 'complete');
  
  return {
    textChunks,
    chartChunks,
    tableChunks,
    hasError,
    isComplete,
    
    /** Get combined text content */
    combinedText: textChunks.map(c => c.content as string).join('\n'),
  };
}
