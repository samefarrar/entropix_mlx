# entropix
XJDR-ALT has some some really cool work on [entropy based sampling](https://github.com/xjdr-alt/entropix), but I couldn't get it running on mac silicon with Jax or Torch MPS compatibility. So I'm trying to fork it to work with MLX.

## Intentions

Entropy Based Sampling and Parallel CoT Decoding

The goal is to use entropy to make context aware sampling. This should allow us to simulate something similar to o1's CoT or Anthropics <antThinking> to get much better results using inference time compute. This project is a research project and a work in process. Its comprised of an inference stack, the sampler, and a UI (future). Please reach out to me on X if you have any question or concerns @_xjdr (original idea and implementation), @samefarrar (MLX implementation).

![Entropy Quadrant](images/entropy_quadrants.png)

## Ethos
Generally, when LLMs pick tokens to output they do so with a set of fixed parameters. You might vary the temperature, the top_k, add logit or repetition penalties but these are fixed for that generation. This means that for every token in response to a question, the way the model samples from the logits is the same.

This doesn't necessarily make sense - some tokens are very straightforward, whereas some tokens might benefit from different sampling to scale inference time compute. As a concrete example, when you ask a model to compare 9.9 or 9.11, the token "." is very "certain". Everywhere in the response to the question, " 9" will likely be followed by ".". Here, scaling inference time compute is wasted because the most likely token is definitely the right one. This is a perfect example of a token where argmax of the logits makes sense.

However, there are tokens that are less "clear", and we think that we can detect this through statistics of the distribution of the logits and the attention scores. For example:


<img width="400" alt="Screenshot 2024-10-12 at 10 34 06" src="https://github.com/user-attachments/assets/79bc42c3-99a9-48d5-bf49-21954292b6c6"> <img width="400" alt="Screenshot 2024-10-12 at 10 41 49" src="https://github.com/user-attachments/assets/aebe1540-ed66-43e2-b099-9b71f30c9daa">

We can see in this example that "compare" is acting as a kind of "uncertainty sink", it is a token that is sampled where variance of the logits varentropy is quite high. In order to scale inference time compute, in the above quadrants, this would be a token well suited to branching. So for now we sample that at that token with a high temperature to try to prevent the model from answering quickly, wrongly and confidently, instead to mimic chain of thought thinking to make it more likely to come to the right answer.

Current supported models:
  llama3.1+

# TODOS:
- Clean up UI (make it look nicer)
- Introduce frog branch sampling parameters
- Allow comparison of metrics from multiple timesteps - we see that attention entropy gradually increases as the model comes to a "decision phrase" e.g. "9.9 is ".

# Getting Started
[install uv](https://github.com/astral-sh/uv)

[install bun if you want to use the local server](https://bun.sh/docs/installation)

```bash
uv sync
```

download weights (Instruct), you need to have **[set up your huggingface cli](https://huggingface.co/docs/huggingface_hub/guides/cli#getting-started) for this!**
```
uv run mlx_download_weights.py
```

## If you want to run the command line prompter:
```bash
uv run mlx_main.py
```
### Options
- `--prompts`: Use predefined prompts from `mlx_entropix.prompts`
- `--prompt_csv`: Use prompts from `data/prompts.csv`
- `--input TEXT`: Provide a custom input prompt
- `--normal`: Use default MLX Llama for generation

## If you want to run the server
```bash
cd ui
bun run dev
```
This will call `uv run mlx_server.py` in the background, as well as the web server.

`--normal`: Use normal model for generation (as opposed to the entropix model)

### Functionality
1. **Model Loading**:
   - Loads either a standard language model or an Entropix model based on the specified options.
   - Uses the Llama-3.2-1B-Instruct model by default.

2. **Text Generation**:
   - Generates text using either the mlx_lm `generate_mlx_lm` function or the Entropix `generate` function.
   - Supports a maximum token limit of 4096.

3. **Command line or Server**
   - Use the model with the command line or the server.

### Examples
1. Use predefined prompts:
   ```
   uv run mlx_main.py --prompts
   ```

2. Use a custom input:
   ```
   uv run mlx_main.py --input "What is the capital of France?"
   ```

3. Use normal sampling instead of Entropix:
   ```
   uv run mlx_main.py --normal --input "Explain quantum computing"
   ```

### Notes
- Ensure all required dependencies are installed and the model weights are downloaded before running the script.
- The Entropix model is used by default unless the `--normal` flag is specified.
