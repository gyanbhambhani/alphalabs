'use client';

import { useEffect, useRef } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import type { DailySnapshot } from '@/types';

interface PortfolioChartProps {
  snapshots: Record<string, DailySnapshot[]>;
  managerNames: Record<string, string>;
}

const COLORS = [
  '#3b82f6', // blue - GPT
  '#8b5cf6', // purple - Claude
  '#f59e0b', // amber - Gemini
  '#22c55e', // green - Quant Bot
];

export function PortfolioChart({ snapshots, managerNames }: PortfolioChartProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    // Get canvas dimensions
    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);
    
    const width = rect.width;
    const height = rect.height;
    const padding = { top: 20, right: 80, bottom: 30, left: 60 };
    const chartWidth = width - padding.left - padding.right;
    const chartHeight = height - padding.top - padding.bottom;
    
    // Clear canvas
    ctx.clearRect(0, 0, width, height);
    
    // Get all manager IDs
    const managerIds = Object.keys(snapshots);
    if (managerIds.length === 0) {
      ctx.fillStyle = '#666';
      ctx.font = '14px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('No performance data yet', width / 2, height / 2);
      return;
    }
    
    // Find data range
    let minReturn = 0;
    let maxReturn = 0;
    let maxDays = 0;
    
    managerIds.forEach((id) => {
      const data = snapshots[id];
      if (data.length > maxDays) maxDays = data.length;
      data.forEach((d) => {
        if (d.cumulativeReturn < minReturn) minReturn = d.cumulativeReturn;
        if (d.cumulativeReturn > maxReturn) maxReturn = d.cumulativeReturn;
      });
    });
    
    // Add padding to range
    const range = maxReturn - minReturn;
    minReturn -= range * 0.1;
    maxReturn += range * 0.1;
    if (minReturn > -0.05) minReturn = -0.05;
    if (maxReturn < 0.05) maxReturn = 0.05;
    
    // Helper functions
    const xScale = (i: number) => padding.left + (i / Math.max(maxDays - 1, 1)) * chartWidth;
    const yScale = (v: number) => 
      padding.top + chartHeight - ((v - minReturn) / (maxReturn - minReturn)) * chartHeight;
    
    // Draw grid
    ctx.strokeStyle = '#333';
    ctx.lineWidth = 0.5;
    
    // Horizontal grid lines
    const yTicks = 5;
    for (let i = 0; i <= yTicks; i++) {
      const y = padding.top + (i / yTicks) * chartHeight;
      ctx.beginPath();
      ctx.moveTo(padding.left, y);
      ctx.lineTo(width - padding.right, y);
      ctx.stroke();
      
      // Y-axis labels
      const value = maxReturn - (i / yTicks) * (maxReturn - minReturn);
      ctx.fillStyle = '#666';
      ctx.font = '10px monospace';
      ctx.textAlign = 'right';
      ctx.fillText(
        `${value >= 0 ? '+' : ''}${(value * 100).toFixed(1)}%`,
        padding.left - 5,
        y + 3
      );
    }
    
    // Zero line
    if (minReturn < 0 && maxReturn > 0) {
      const zeroY = yScale(0);
      ctx.strokeStyle = '#666';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(padding.left, zeroY);
      ctx.lineTo(width - padding.right, zeroY);
      ctx.stroke();
    }
    
    // Draw lines for each manager
    managerIds.forEach((id, idx) => {
      const data = snapshots[id];
      if (data.length === 0) return;
      
      const color = COLORS[idx % COLORS.length];
      
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.beginPath();
      
      data.forEach((d, i) => {
        const x = xScale(i);
        const y = yScale(d.cumulativeReturn);
        
        if (i === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      });
      
      ctx.stroke();
      
      // Draw endpoint marker
      const lastPoint = data[data.length - 1];
      const lastX = xScale(data.length - 1);
      const lastY = yScale(lastPoint.cumulativeReturn);
      
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(lastX, lastY, 4, 0, Math.PI * 2);
      ctx.fill();
      
      // Legend
      const legendY = padding.top + idx * 18;
      ctx.fillStyle = color;
      ctx.fillRect(width - padding.right + 10, legendY, 12, 12);
      ctx.fillStyle = '#999';
      ctx.font = '11px sans-serif';
      ctx.textAlign = 'left';
      ctx.fillText(
        managerNames[id] || id,
        width - padding.right + 26,
        legendY + 10
      );
    });
  }, [snapshots, managerNames]);
  
  return (
    <Card>
      <CardHeader>
        <CardTitle>Performance Over Time</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="h-64 w-full">
          <canvas
            ref={canvasRef}
            className="w-full h-full"
            style={{ width: '100%', height: '100%' }}
          />
        </div>
      </CardContent>
    </Card>
  );
}
