![Mirix Logo](https://github.com/RenKoya1/MIRIX/raw/main/assets/logo.png)

## MIRIX - Multi-Agent Personal Assistant with an Advanced Memory System

Your personal AI that builds memory through screen observation and natural conversation

<table>
  <tr>
    <td style="border-left: 6px solid #d35400; background: #fff3e0; padding: 12px;">
      <strong>Important Update: 0.1.6 (Main) vs 0.1.3 (Desktop Agent)</strong><br/>
      Starting with <code>0.1.6</code>, the <code>main</code> branch is a brand-new release line where Mirix is a pure memory system that can be plugged into any existing agents. The desktop personal assistant (frontend + backend) has been deprecated and is no longer shipped on <code>main</code>. If you need the earlier desktop application with the built-in agent, use the <code>desktop-agent</code> branch.
    </td>
  </tr>
</table>

| 🌐 [Website](https://mirix.io) | 📚 [Documentation](https://docs.mirix.io) | 📄 [Paper](https://arxiv.org/abs/2507.07957) | 💬 [Discord](https://discord.gg/S6CeHNrJ) 
<!-- | [Twitter/X](https://twitter.com/mirix_ai) | [Discord](https://discord.gg/S6CeHNrJ) | -->

---

### Key Features 🔥

- **Multi-Agent Memory System:** Six specialized memory components (Core, Episodic, Semantic, Procedural, Resource, Knowledge) managed by dedicated agents
- **Screen Activity Tracking:** Continuous visual data capture and intelligent consolidation into structured memories  
- **Privacy-First Design:** All long-term data stored locally with user-controlled privacy settings
- **Advanced Search:** PostgreSQL-native BM25 full-text search with vector similarity support
- **Multi-Modal Input:** Text, images, voice, and screen captures processed seamlessly

### Quick Start
**Option A: Cloud (hosted API):**
```
pip install mirix==0.1.6
```
Get your API key and view memory call traces at https://app.mirix.io.
```python
from mirix import MirixClient

client = MirixClient(api_key="your_api_key_here")
# or set MIRIX_API_KEY in your environment, then use: client = MirixClient()

client.initialize_meta_agent(
    provider="openai"
)  # See configs in mirix/configs/examples/mirix_openai.yaml

# Simple add example
client.add(
    user_id="demo-user",
    messages=[
        {"role": "user", "content": [{"type": "text", "text": "The moon now has a president."}]},
        {"role": "assistant", "content": [{"type": "text", "text": "Noted."}]},
    ],
)

# For a full example, see README.md below or samples/run_client.py
```

**Option B: Local (backend + dashboard, no Docker):**
**Step 1: Backend & Dashboard**
```
pip install -r requirements.txt
```
In terminal 1:
```
python scripts/start_server.py
```
In terminal 2:
```
cd dashboard
npm install
npm run dev
```
- Dashboard: http://localhost:5173  
- API: http://localhost:8531  

**Step 2: Create an API key in the dashboard (http://localhost:5173) and set as the environmental variable `MIRIX_API_KEY`.**

> **⚠️ Important Note for Local Development:**  
> When using the local version, **do NOT use the PyPI package** (`mirix==0.1.6`). You must run your code from the root folder of this project so that the local `mirix` module can be properly imported. If you previously installed the PyPI package, uninstall it first:
> ```bash
> pip uninstall mirix
> ```

Now you are ready to go! See the example below:
```python
from mirix import MirixClient

client = MirixClient(
    api_key="your-api-key", # if you set MIRIX_API_KEY as the environmental variable then you don't need this
    base_url="http://localhost:8531",
)

client.initialize_meta_agent(
    config={
        "llm_config": {
            "model": "gpt-4o-mini",
            "model_endpoint_type": "openai",
            "model_endpoint": "https://api.openai.com/v1",
            "context_window": 128000,
        },
        "build_embeddings_for_memory": True,
        "embedding_config": {
            "embedding_model": "text-embedding-3-small",
            "embedding_endpoint": "https://api.openai.com/v1",
            "embedding_endpoint_type": "openai",
            "embedding_dim": 1536,
        },
        "meta_agent_config": {
            "agents": [
                "core_memory_agent",
                "resource_memory_agent",
                "semantic_memory_agent",
                "episodic_memory_agent",
                "procedural_memory_agent",
                "knowledge_memory_agent",
                "reflexion_agent",
                "background_agent",
            ],
            "memory": {
                "core": [
                    {"label": "human", "value": ""},
                    {"label": "persona", "value": "I am a helpful assistant."},
                ],
                "decay": {
                    "fade_after_days": 30,
                    "expire_after_days": 90,
                },
            },
        },
    }
)

client.add(
    user_id="demo-user",
    messages=[
        {"role": "user", "content": [{"type": "text", "text": "The moon now has a president."}]},
        {"role": "assistant", "content": [{"type": "text", "text": "Noted."}]},
    ],
)

memories = client.retrieve_with_conversation(
    user_id="demo-user",
    messages=[
        {"role": "user", "content": [{"type": "text", "text": "What did we discuss on MirixDB in last 4 days?"}]},
    ],
    limit=5,
)
print(memories)
```
For more API examples, see `samples/run_client.py`.

## License

Mirix is released under the Apache License 2.0. See the [LICENSE](LICENSE) file for more details.

## Contact

For questions, suggestions, or issues, please open an issue on the GitHub repository or contact us at `founders@mirix.io`

## Join Our Community

Connect with other Mirix users, share your thoughts, and get support:

### 💬 Discord Community
Join our Discord server for real-time discussions, support, and community updates:
**[https://discord.gg/FXtXJuRf](https://discord.gg/FXtXJuRf)**

### 🎯 Weekly Discussion Sessions
We host weekly discussion sessions where you can:
- Discuss issues and bugs
- Share ideas about future directions
- Get general consultations and support
- Connect with the development team and community

**📅 Schedule:** Friday nights, 8-9 PM PST  
**🔗 Zoom Link:** [https://ucsd.zoom.us/j/96278791276](https://ucsd.zoom.us/j/96278791276)

### 📱 WeChat Group
You can add the account `ari_asm` so that I can add you to the group chat.

## Acknowledgement
We would like to thank [Letta](https://github.com/letta-ai/letta) for open-sourcing their framework, which served as the foundation for the memory system in this project.
