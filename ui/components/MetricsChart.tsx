"use client";

import React from "react";
import {
  CartesianGrid,
  Line,
  LineChart,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
} from "recharts";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  ChartConfig,
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
} from "@/components/ui/chart";
import { Metric } from "@/types/chat";

interface MetricsChartProps {
  metrics: Metric[];
}

const chartConfig = {
  logits_entropy: {
    label: "Logits Entropy",
    color: "hsl(var(--chart-1))",
  },
  logits_varentropy: {
    label: "Logits Varentropy",
    color: "hsl(var(--chart-2))",
  },
  attention_entropy: {
    label: "Attention Entropy",
    color: "hsl(var(--chart-3))",
  },
  attention_varentropy: {
    label: "Attention Varentropy",
    color: "hsl(var(--chart-4))",
  },
  agreement: {
    label: "Agreement",
    color: "hsl(var(--chart-5))",
  },
  interaction_strength: {
    label: "Interaction Strength",
    color: "hsl(var(--chart-6))",
  },
} satisfies ChartConfig;

export function MetricsChart({ metrics }: MetricsChartProps) {
  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle>Generation Metrics</CardTitle>
        <CardDescription>Metrics for the generated response</CardDescription>
      </CardHeader>
      <CardContent>
        <ChartContainer config={chartConfig}>
          <LineChart
            data={metrics}
            margin={{
              left: 12,
              right: 12,
            }}
          >
            <CartesianGrid vertical={false} />
            <XAxis
              dataKey="cur_pos"
              tickLine={false}
              axisLine={false}
              tickMargin={8}
            />
            <YAxis />
            <ChartTooltip cursor={false} content={<ChartTooltipContent />} />
            {Object.entries(chartConfig).map(([key, config]) => (
              <Line
                key={key}
                type="monotone"
                dataKey={key}
                stroke={config.color}
                strokeWidth={2}
                dot={false}
              />
            ))}
          </LineChart>
        </ChartContainer>
      </CardContent>
      <CardFooter>
        <div className="text-sm text-muted-foreground">
          Showing metrics for each token position in the generated response
        </div>
      </CardFooter>
    </Card>
  );
}
