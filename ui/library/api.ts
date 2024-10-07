// @/library/api.ts
import { Message } from "@/types/chat";
const API_URL = "http://localhost:8000"; // Update this with your server's URL

export async function sendMessage(
  messages: Message[],
  modelId: string,
  onUpdate?: (update: string) => void,
) {

  const response = await fetch(`${API_URL}/v1/chat/completions`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model: modelId,
      messages: messages.map((msg) => ({
        role: msg.role,
        content: msg.role === "assistant" ? JSON.parse(msg.content).response : msg.content,
      })),
      stream: !!onUpdate,
    }),
  });

  if (!response.ok) {
    throw new Error("Failed to send message");
  }

  if (onUpdate) {
    // Streaming case
    const reader = response.body?.getReader();
    const decoder = new TextDecoder();
    let fullResponse = "";

    while (true) {
      const { done, value } = await reader?.read();
      if (done) break;

      const chunk = decoder.decode(value);
      const lines = chunk.split("\n");

      for (const line of lines) {
        if (line.startsWith("data: ")) {
          const data = line.slice(6);
          if (data === "[DONE]") break;
          try {
            const parsed = JSON.parse(data);
            const content = parsed.choices[0]?.delta?.content || "";
            fullResponse += content;
            onUpdate(fullResponse);
          } catch (error) {
            console.error("Error parsing JSON:", error);
          }
        }
      }
    }

    return JSON.stringify({
      response: fullResponse,
      thinking: "Thinking process...",
      user_mood: "neutral",
    });
  } else {
    // Non-streaming case
    const api_response = response;
    console.log("api_response", api_response);
    return new Promise<string>((resolve) => {
      setTimeout(() => {
        resolve(
          JSON.stringify({
            response: api_response.choices[0].message.content,
            thinking: "Thinking process...",
            user_mood: "neutral",
          }),
        );
      }, 1000);
    });
  }
}
