import React from "react";
import {
  CartesianGrid,
  Line,
  LineChart,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  LabelList,
  Label,
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
import { Skeleton } from "@/components/ui/skeleton";

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
  token: {
    label: "Token",
  },
} satisfies ChartConfig;

export function MetricsChart({ metrics }: MetricsChartProps) {
  if (!metrics || metrics.length === 0) {
    return (
      <Card className="w-full">
        <CardHeader>
          <CardTitle>Generation Metrics</CardTitle>
          <CardDescription>Loading metrics...</CardDescription>
        </CardHeader>
        <CardContent>
          <Skeleton className="h-[200px] w-full" />
        </CardContent>
      </Card>
    );
  }
  console.log(metrics);
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
              top: 30, // Increased top margin to accommodate the title
              left: 12,
              right: 12,
              bottom: 5,
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
            <ChartTooltip
              cursor={false}
              content={
                <ChartTooltipContent
                  labelFormatter={(label, payload) => {
                    const token = payload[0]?.payload?.token;
                    return `Token: ${token}`;
                  }}
                />
              }
            />
            {Object.entries(chartConfig).map(([key, config]) => {
              if (key !== "token") {
                return (
                  <Line
                    key={key}
                    type="monotone"
                    dataKey={key}
                    stroke={config.color}
                    strokeWidth={2}
                    dot={false}
                  />
                );
              }
              return null;
            })}
          </LineChart>
        </ChartContainer>
      </CardContent>
    </Card>
  );
}
