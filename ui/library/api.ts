// @/library/api.ts
import { Message } from "./types";
const API_URL = "http://localhost:8000"; // Update this with your server's URL
const MAX_TOKENS = 2048;

export async function sendMessage(
  message: Message[] | Message,
  systemPrompt: string,
  modelId: string,
  onUpdate?: (update: string) => void,
) {
  console.log("Pre-parsed messages: ", message);
  const messages = Array.isArray(message)
    ? message.map(({ role, content }) => ({ role, content }))
    : [{ role: "user", content: message }];

  let processedMessages = messages.map((message) => {
    if (message.role === "assistant" && message.content) {
      try {
        const parsedContent = JSON.parse(message.content);
        return { ...message, content: parsedContent.response };
      } catch (error) {
        console.error("Error parsing assistant message content:", error);
        return message;
      }
    }
    return message;
  });

  // Only add the system prompt if it's not empty
  if (systemPrompt.trim() !== "") {
    processedMessages = [
      { role: "system", content: systemPrompt },
      ...processedMessages,
    ];
  }

  console.log("Sending messages:", processedMessages);
  const response = await fetch(`${API_URL}/v1/chat/completions`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model: modelId,
      messages: processedMessages.map(({ role, content }) => ({
        role,
        content,
      })),
      max_tokens: MAX_TOKENS,
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
    return fullResponse; // Return the full response string directly
  } else {
    // Non-streaming case
    const api_response = await response.json();
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
