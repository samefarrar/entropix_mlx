export interface Message {
  id: string;
  role: "user" | "assistant" | "system";
  content: string;
  metrics?: MessageMetrics;
}

export interface Metric {
  cur_pos: number;
  logits_entropy: number;
  logits_varentropy: number;
  attention_entropy: number;
  attention_varentropy: number;
  agreement: number;
  interaction_strength: number;
}

export interface MessageMetrics {
  metrics: Metric[];
}

export type Model = {
  id: string;
  name: string;
};

export interface ArtifactContent {
  id: string;
  type: "code" | "html" | "text" | "log";
  content: string;
  language?: string;
  name: string;
}

export interface ParsedResponse {
  response: string;
}

export interface MessageContentState {
  thinking: boolean;
  parsed: ParsedResponse;
  error: boolean;
}
