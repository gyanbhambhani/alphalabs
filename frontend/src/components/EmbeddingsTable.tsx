'use client';

import { useState } from 'react';
import { ChevronLeft, ChevronRight, ChevronsLeft, ChevronsRight, ArrowUpDown } from 'lucide-react';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import { Button } from '@/components/ui/button';
import type { Embedding } from '@/types';
import { cn } from '@/lib/utils';

interface EmbeddingsTableProps {
  embeddings: Embedding[];
  total: number;
  page: number;
  perPage: number;
  onPageChange: (page: number) => void;
  onSort: (field: string, order: 'asc' | 'desc') => void;
  onRowClick?: (embedding: Embedding) => void;
  sortBy?: string;
  sortOrder?: 'asc' | 'desc';
  isLoading?: boolean;
}

export function EmbeddingsTable({
  embeddings,
  total,
  page,
  perPage,
  onPageChange,
  onSort,
  onRowClick,
  sortBy = 'date',
  sortOrder = 'desc',
  isLoading = false,
}: EmbeddingsTableProps) {
  const totalPages = Math.ceil(total / perPage);

  const handleSort = (field: string) => {
    const newOrder = 
      sortBy === field && sortOrder === 'desc' ? 'asc' : 'desc';
    onSort(field, newOrder);
  };

  const SortIcon = ({ field }: { field: string }) => {
    if (sortBy !== field) {
      return <ArrowUpDown className="ml-2 h-4 w-4 opacity-30" />;
    }
    return (
      <ArrowUpDown 
        className={cn(
          "ml-2 h-4 w-4",
          sortOrder === 'desc' ? 'rotate-180' : ''
        )} 
      />
    );
  };

  const formatPercent = (value: number) => {
    const pct = value * 100;
    const color = value >= 0 ? 'text-green-500' : 'text-red-500';
    return (
      <span className={color}>
        {value >= 0 ? '+' : ''}{pct.toFixed(2)}%
      </span>
    );
  };

  if (isLoading) {
    return (
      <div className="text-center py-12 text-muted-foreground">
        Loading embeddings...
      </div>
    );
  }

  if (embeddings.length === 0) {
    return (
      <div className="text-center py-12 text-muted-foreground">
        No embeddings found. Try a different search query.
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="rounded-md border">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>
                <button
                  onClick={() => handleSort('date')}
                  className="flex items-center font-semibold hover:text-foreground"
                >
                  Date
                  <SortIcon field="date" />
                </button>
              </TableHead>
              <TableHead className="text-right">
                <button
                  onClick={() => handleSort('price')}
                  className="flex items-center justify-end w-full 
                             font-semibold hover:text-foreground"
                >
                  Price
                  <SortIcon field="price" />
                </button>
              </TableHead>
              <TableHead className="text-right">
                <button
                  onClick={() => handleSort('return_1m')}
                  className="flex items-center justify-end w-full 
                             font-semibold hover:text-foreground"
                >
                  1M Return
                  <SortIcon field="return_1m" />
                </button>
              </TableHead>
              <TableHead className="text-right">3M Return</TableHead>
              <TableHead className="text-right">6M Return</TableHead>
              <TableHead className="text-right">
                <button
                  onClick={() => handleSort('volatility_21d')}
                  className="flex items-center justify-end w-full 
                             font-semibold hover:text-foreground"
                >
                  21D Vol
                  <SortIcon field="volatility_21d" />
                </button>
              </TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {embeddings.map((embedding) => (
              <TableRow
                key={embedding.id}
                onClick={() => onRowClick?.(embedding)}
                className={cn(
                  onRowClick && "cursor-pointer hover:bg-muted/50"
                )}
              >
                <TableCell className="font-mono">
                  {embedding.metadata.date}
                </TableCell>
                <TableCell className="text-right font-mono">
                  ${embedding.metadata.price.toFixed(2)}
                </TableCell>
                <TableCell className="text-right font-mono">
                  {formatPercent(embedding.metadata.return1m)}
                </TableCell>
                <TableCell className="text-right font-mono">
                  {formatPercent(embedding.metadata.return3m)}
                </TableCell>
                <TableCell className="text-right font-mono">
                  {formatPercent(embedding.metadata.return6m)}
                </TableCell>
                <TableCell className="text-right font-mono">
                  {(embedding.metadata.volatility21d * 100).toFixed(2)}%
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </div>

      {/* Pagination */}
      <div className="flex items-center justify-between">
        <div className="text-sm text-muted-foreground">
          Showing {(page - 1) * perPage + 1} to{' '}
          {Math.min(page * perPage, total)} of {total} embeddings
        </div>

        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => onPageChange(1)}
            disabled={page === 1}
          >
            <ChevronsLeft className="h-4 w-4" />
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={() => onPageChange(page - 1)}
            disabled={page === 1}
          >
            <ChevronLeft className="h-4 w-4" />
          </Button>

          <div className="text-sm">
            Page {page} of {totalPages}
          </div>

          <Button
            variant="outline"
            size="sm"
            onClick={() => onPageChange(page + 1)}
            disabled={page === totalPages}
          >
            <ChevronRight className="h-4 w-4" />
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={() => onPageChange(totalPages)}
            disabled={page === totalPages}
          >
            <ChevronsRight className="h-4 w-4" />
          </Button>
        </div>
      </div>
    </div>
  );
}
