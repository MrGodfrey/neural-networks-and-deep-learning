# Local assets for GPT lecture demo

- Mechanism model: `sshleifer/tiny-gpt2`, used for token/logits/attention demonstrations.
- Assistant model: `Qwen/Qwen2.5-0.5B-Instruct`, downloaded on demand to `qwen2_5_0_5b_instruct/` and used for real generated assistant answers.
- Data: first 80 examples extracted from `roneneldan/TinyStories/TinyStories-valid.txt`.
- Purpose: classroom demonstration only; this is not a representative training set.

The notebook first checks local files. Missing assets are downloaded on demand when `DOWNLOAD_MISSING_ASSETS=True`; after files exist, the notebook loads them from local paths.

Download behavior:

- Running section 0 of `GPT课堂演示_离线版.ipynb` downloads the tiny GPT-2 mechanism model and TinyStories sample if missing.
- Running section 3.1, or the first call to `load_assistant_model()` / `chat_generate()`, downloads Qwen2.5-0.5B-Instruct if missing.
- To predownload Qwen before class, run section 0 and then run section 3.1 or `load_assistant_model()` once while online.

Large downloaded model directories should stay out of git tracking. In this repo, `qwen2_5_0_5b_instruct/` is ignored by `.gitignore`.
