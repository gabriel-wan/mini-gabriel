# TODO.md — roadmap

Concrete next steps, grouped by area. See [PROJECT.md](PROJECT.md) for context and open decisions.

## Data

Tooling — implemented:

- [x] Configure Telegram API access
- [x] Implement Telethon extraction
- [x] Implement the chat-analysis report

Execution:

- [x] Extract 2026 messages
- [x] Analyze candidate chats
- [x] Inspect dataset quality
- [x] Finalize filtering criteria
- [x] Implement training-example construction

## ML

Tooling — implemented:

- [x] Implement the style-similarity evaluation
- [x] Select fine-tuning method (LoRA, see CLUSTER.md)

Execution:

- [ ] Select baseline model
- [ ] Establish baseline prompting/inference
- [ ] Run initial fine-tuning experiment
- [ ] Evaluate style similarity against the baseline
- [ ] Iterate

## Application

- [ ] Build chatbot interface
- [ ] Connect fine-tuned model
- [ ] Test conversational quality
