import React, { useState } from "react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { X } from "lucide-react";

interface SystemPromptSelectorProps {
  systemPrompt: string;
  onSystemPromptChange: (prompt: string) => void;
}

const SystemPromptSelector: React.FC<SystemPromptSelectorProps> = ({
  systemPrompt,
  onSystemPromptChange,
}) => {
  const [editedPrompt, setEditedPrompt] = useState(systemPrompt);

  const handlePromptChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setEditedPrompt(e.target.value);
  };

  const handleSave = () => {
    onSystemPromptChange(editedPrompt);
  };

  const handleClear = () => {
    setEditedPrompt("");
    onSystemPromptChange("");
  };

  return (
    <div className="mt-auto">
      <h3 className="text-sm font-semibold mb-2">System Prompt</h3>
      <Textarea
        value={editedPrompt}
        onChange={handlePromptChange}
        className="min-h-[100px] mb-2"
        placeholder="Enter system prompt..."
      />
      <div className="flex justify-between">
        <Button onClick={handleSave} size="sm">
          Save
        </Button>
        <Button onClick={handleClear} size="sm" variant="outline">
          <X className="h-4 w-4 mr-2" />
          Clear
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
