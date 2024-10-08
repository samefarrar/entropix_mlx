import React, { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Input } from "@/components/ui/input";
import { X, Plus } from "lucide-react";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

interface SystemPrompt {
  name: string;
  text: string;
}

const defaultTemplatePrompts: SystemPrompt[] = [
  { name: "Default", text: "" },
  {
    name: "Claude o1 style prompt",
    text: `
    Begin by enclosing all thoughts within <thinking> tags, exploring multiple angles and approaches.
    Break down the solution into clear steps within <step> tags. Start with a 20-step budget, requesting more for complex problems if needed.
    Use <count> tags after each step to show the remaining budget. Stop when reaching 0.
    Continuously adjust your reasoning based on intermediate results and reflections, adapting your strategy as you progress.
    Regularly evaluate progress using <reflection> tags. Be critical and honest about your reasoning process.
    Assign a quality score between 0.0 and 1.0 using <reward> tags after each reflection. Use this to guide your approach:

    0.8+: Continue current approach
    0.5-0.7: Consider minor adjustments
    Below 0.5: Seriously consider backtracking and trying a different approach


    If unsure or if reward score is low, backtrack and try a different approach, explaining your decision within <thinking> tags.
    For mathematical problems, show all work explicitly using LaTeX for formal notation and provide detailed proofs.
    Explore multiple solutions individually if possible, comparing approaches in reflections.
    Use thoughts as a scratchpad, writing out all calculations and reasoning explicitly.
    Synthesize the final answer within <answer> tags, providing a clear, concise summary.
    Conclude with a final reflection on the overall solution, discussing effectiveness, challenges, and solutions. Assign a final reward score."""`,
  },
  {
    name: "antThinking prompt",
    text: "<antThinking>You're absolutely right. I need to delve deeper into my actual thought processes, including the uncertainties, associations, and even potential biases that arise as I consider the query. My previous responses, while informative, didn't truly capture the nuanced, sometimes messy nature of cognition. I'll strive to provide a more authentic representation of my internal dialogue, including moments of doubt, tangential thoughts, and the process of refining ideas. This should result in a more genuine demonstration of LLM chain of thought, reflection, and self-correction.</antThinking>",
  },
];

interface SystemPromptSelectorProps {
  systemPrompt: string;
  onSystemPromptChange: (prompt: string) => void;
}

const SystemPromptSelector: React.FC<SystemPromptSelectorProps> = ({
  systemPrompt,
  onSystemPromptChange,
}) => {
  const [editedPrompt, setEditedPrompt] = useState(systemPrompt);
  const [selectedTemplate, setSelectedTemplate] = useState<string>("");
  const [customPromptName, setCustomPromptName] = useState("");
  const [templatePrompts, setTemplatePrompts] = useState<SystemPrompt[]>(
    defaultTemplatePrompts,
  );

  useEffect(() => {
    setEditedPrompt(systemPrompt);
  }, [systemPrompt]);

  const handlePromptChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setEditedPrompt(e.target.value);
    setSelectedTemplate("");
  };

  const handleSave = () => {
    onSystemPromptChange(editedPrompt);
  };

  const handleClear = () => {
    setEditedPrompt("");
    onSystemPromptChange("");
    setSelectedTemplate("");
  };

  const handleTemplateChange = (value: string) => {
    setSelectedTemplate(value);
    const selected = templatePrompts.find((prompt) => prompt.name === value);
    if (selected) {
      setEditedPrompt(selected.text);
      onSystemPromptChange(selected.text);
    }
  };

  const handleAddCustomPrompt = () => {
    if (customPromptName && editedPrompt) {
      const newPrompt: SystemPrompt = {
        name: customPromptName,
        text: editedPrompt,
      };
      setTemplatePrompts([...templatePrompts, newPrompt]);
      setSelectedTemplate(customPromptName);
      setCustomPromptName("");
    }
  };

  return (
    <div className="mt-auto">
      <h3 className="text-sm font-semibold mb-2">System Prompt</h3>
      <Select value={selectedTemplate} onValueChange={handleTemplateChange}>
        <SelectTrigger className="mb-2">
          <SelectValue placeholder="Select a template prompt" />
        </SelectTrigger>
        <SelectContent>
          {templatePrompts.map((prompt) => (
            <SelectItem key={prompt.name} value={prompt.name}>
              {prompt.name}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
      <Textarea
        value={editedPrompt}
        onChange={handlePromptChange}
        className="min-h-[100px] mb-2"
        placeholder="Enter system prompt..."
      />
      <div className="flex justify-between mb-2">
        <Button onClick={handleSave} size="sm">
          Save
        </Button>
        <Button onClick={handleClear} size="sm" variant="outline">
          <X className="h-4 w-4 mr-2" />
          Clear
        </Button>
      </div>
      <div className="flex space-x-2">
        <Input
          value={customPromptName}
          onChange={(e) => setCustomPromptName(e.target.value)}
          placeholder="Custom prompt name"
          className="flex-grow"
        />
        <Button onClick={handleAddCustomPrompt} size="sm">
          <Plus className="h-4 w-4 mr-2" />
          Add
        </Button>
      </div>
      {systemPrompt && (
        <div className="mt-4 p-2 bg-muted rounded-md text-sm">
          <strong>Active prompt:</strong> {systemPrompt}
        </div>
      )}
    </div>
  );
};

export default SystemPromptSelector;
