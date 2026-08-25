# TODO.md — roadmap

Concrete next steps, grouped by area. See [PROJECT.md](PROJECT.md) for context and open decisions.

## Data

Tooling — implemented:

- [x] Configure Telegram API access
- [x] Implement Telethon extraction
- [x] Implement the chat-analysis report

Execution — needs a real run against the account:

- [ ] Extract 2026 messages
- [ ] Analyze candidate chats
- [ ] Inspect dataset quality
- [ ] Finalize filtering criteria
- [ ] Construct training examples

## ML

- [ ] Select baseline model
- [ ] Establish baseline prompting/inference
- [ ] Select fine-tuning method
- [ ] Run initial fine-tuning experiment
- [ ] Evaluate style similarity
- [ ] Iterate

## Application

- [ ] Build chatbot interface
- [ ] Connect fine-tuned model
- [ ] Test conversational quality
