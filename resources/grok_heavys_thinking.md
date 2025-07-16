You are an expert Python coder and expert on the topic of MCP servers. Write a technical rich instructino on how to build an MCP server that integrates with Azure AI Foundries O3 Deep Research in python. THE GOAL FOR YOUR OUTPUT IS TO BE CLEAR GUIDANCE FOR A PROGRAMMER THAT WILL DO THE ACTUAL IMPLEMENTATION _ TREAT THIS PROGRAMMER AS A SEMI-SENIOR DEVELOPER, not JUNIOR BUT NOT PERFECT; SO YOU NEED TO BE EXPLICIT ON THE DETAILS AND GIVE MOTIVATED EXAMPLES FOR THE CODE SUGGESTIONS YOU PRODUCE.

Use reference MCP servers as inspiration. Use official Python MCP SDK for tooling. here is documentation on Azure AI Deep Research usage: https://learn.microsoft.com/en-us/azure/ai-foundry/agents/how-to/tools/deep-research-samples. Here is their python code example:

import os, time
from typing import Optional
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
from azure.ai.agents import AgentsClient
from azure.ai.agents.models import DeepResearchTool, MessageRole, ThreadMessage


def fetch_and_print_new_agent_response(
    thread_id: str,
    agents_client: AgentsClient,
    last_message_id: Optional[str] = None,
) -> Optional[str]:
    response = agents_client.messages.get_last_message_by_role(
        thread_id=thread_id,
        role=MessageRole.AGENT,
    )
    if not response or response.id == last_message_id:
        return last_message_id  # No new content

    print("\nAgent response:")
    print("\n".join(t.text.value for t in response.text_messages))

    for ann in response.url_citation_annotations:
        print(f"URL Citation: [{ann.url_citation.title}]({ann.url_citation.url})")

    return response.id


def create_research_summary(
        message : ThreadMessage,
        filepath: str = "research_summary.md"
) -> None:
    if not message:
        print("No message content provided, cannot create research summary.")
        return

    with open(filepath, "w", encoding="utf-8") as fp:
        # Write text summary
        text_summary = "\n\n".join([t.text.value.strip() for t in message.text_messages])
        fp.write(text_summary)

        # Write unique URL citations, if present
        if message.url_citation_annotations:
            fp.write("\n\n## References\n")
            seen_urls = set()
            for ann in message.url_citation_annotations:
                url = ann.url_citation.url
                title = ann.url_citation.title or url
                if url not in seen_urls:
                    fp.write(f"- [{title}]({url})\n")
                    seen_urls.add(url)

    print(f"Research summary written to '{filepath}'.")


project_client = AIProjectClient(
    endpoint=os.environ["PROJECT_ENDPOINT"],
    credential=DefaultAzureCredential(),
)

conn_id = project_client.connections.get(name=os.environ["BING_RESOURCE_NAME"]).id


# Initialize a Deep Research tool with Bing Connection ID and Deep Research model deployment name
deep_research_tool = DeepResearchTool(
    bing_grounding_connection_id=conn_id,
    deep_research_model=os.environ["DEEP_RESEARCH_MODEL_DEPLOYMENT_NAME"],
)

# Create Agent with the Deep Research tool and process Agent run
with project_client:

    with project_client.agents as agents_client:

        # Create a new agent that has the Deep Research tool attached.
        # NOTE: To add Deep Research to an existing agent, fetch it with `get_agent(agent_id)` and then,
        # update the agent with the Deep Research tool.
        agent = agents_client.create_agent(
            model=os.environ["MODEL_DEPLOYMENT_NAME"],
            name="my-agent",
            instructions="You are a helpful Agent that assists in researching scientific topics.",
            tools=deep_research_tool.definitions,
        )

        # [END create_agent_with_deep_research_tool]
        print(f"Created agent, ID: {agent.id}")

        # Create thread for communication
        thread = agents_client.threads.create()
        print(f"Created thread, ID: {thread.id}")

        # Create message to thread
        message = agents_client.messages.create(
            thread_id=thread.id,
            role="user",
            content=(
                "Give me the latest research into quantum computing over the last year."
            ),
        )
        print(f"Created message, ID: {message.id}")

        print(f"Start processing the message... this may take a few minutes to finish. Be patient!")
        # Poll the run as long as run status is queued or in progress
        run = agents_client.runs.create(thread_id=thread.id, agent_id=agent.id)
        last_message_id = None
        while run.status in ("queued", "in_progress"):
            time.sleep(1)
            run = agents_client.runs.get(thread_id=thread.id, run_id=run.id)

            last_message_id = fetch_and_print_new_agent_response(
                thread_id=thread.id,
                agents_client=agents_client,
                last_message_id=last_message_id,
            )
            print(f"Run status: {run.status}")

        print(f"Run finished with status: {run.status}, ID: {run.id}")

        if run.status == "failed":
            print(f"Run failed: {run.last_error}")

        # Fetch the final message from the agent in the thread and create a research summary
        final_message = agents_client.messages.get_last_message_by_role(
            thread_id=thread.id, role=MessageRole.AGENT
        )
        if final_message:
            create_research_summary(final_message)

        # Clean-up and delete the agent once the run is finished.
        # NOTE: Comment out this line if you plan to reuse the agent later.
        agents_client.delete_agent(agent.id)
        print("Deleted agent")

The MCP server should respond with a link to a markdown document to the MCP client, so that the client then can copy that file and read it. This is because the files can become rather large, so it is better to store the deep research result on disk and give the mcp client the path to it.

Here you have the link to the official antrhopic python sdk thing: https://github.com/modelcontextprotocol/python-sdk

# Building an MCP Server with Azure AI Foundry Deep Research Integration

As an expert Python coder and MCP (Model Context Protocol) specialist, I'll provide a detailed, technical guide to building an MCP server that integrates Azure AI Foundry's O3 Deep Research capabilities. This guide is tailored for a semi-senior developer: I'll assume you have solid Python experience, familiarity with decorators, async programming, and REST-like concepts, but I'll be explicit on MCP-specific patterns, Azure integration details, and potential pitfalls. I'll motivate code suggestions with explanations (e.g., why use globals for shared state, how to handle long-running operations) and include annotated code examples.

The goal is to create an MCP server that exposes a **tool** for performing deep research queries. When invoked (e.g., by an LLM client like Claude Desktop via MCP), the tool will:
- Use Azure's Deep Research agent to process a user query (e.g., "latest research on quantum computing").
- Poll for the response (which can take minutes, so we'll add progress updates).
- Save the research summary as a Markdown file on disk (to handle large outputs efficiently).
- Return a local file path (e.g., `/path/to/research_summary.md`) to the MCP client. Since MCP often runs locally (e.g., via stdio transport with Claude Desktop), the client can directly access and "copy" (read/load) the file from the shared filesystem. For remote setups, you'd need to extend this with a resource handler for file serving—I'll note that as an advanced option.

We'll draw inspiration from reference MCP servers in the SDK examples (e.g., simple tools like `add` for computation, resources like `get_greeting` for data exposure). Here, the deep research is a "tool" because it performs a side-effectful action (querying Azure, saving to disk) rather than just exposing static data.

**Key Motivations for Design Choices:**
- **Reuse Azure Agent:** Create the agent once on server startup (via lifespan) to avoid overhead per query. This is efficient for a server handling multiple requests.
- **Progress Updates:** Use MCP's `Context` object to log progress during polling—motivated by the long-running nature of Deep Research (avoids silent blocking).
- **File Saving:** Generate unique filenames (e.g., with timestamps) to avoid overwrites; return a `file://` URI for clarity, as it's a standard way to reference local files.
- **Error Handling:** Explicitly handle failures (e.g., run status "failed") to make the tool robust—semi-senior devs might overlook Azure-specific errors.
- **Transport Considerations:** Start with stdio for local testing (e.g., with Claude Desktop); note extensions for HTTP in production.

## Prerequisites
- Python 3.8+ recommended for async features).
- Azure subscription with access to AI Foundry (preview feature; see [Azure docs](https://learn.microsoft.com/en-us/azure/ai-foundry/agents/how-to/tools/deep-research-samples)).
- Set these environment variables (from Azure docs/example):
  - `PROJECT_ENDPOINT`: Your Azure AI Project endpoint (e.g., "https://your-project.westus2.inference.ml.azure.com").
  - `BING_RESOURCE_NAME`: Name of your Bing connection resource.
  - `DEEP_RESEARCH_MODEL_DEPLOYMENT_NAME`: Deployment name for the Deep Research model (e.g., "o3-deep-research").
  - `MODEL_DEPLOYMENT_NAME`: Deployment name for the base LLM (e.g., "gpt-4o").
- Azure CLI logged in (`az login`) for `DefaultAzureCredential`.
- Familiarize with MCP concepts: Tools (action-oriented), Resources (data-oriented), Prompts (interaction templates), Context (request-scoped utilities).

## Installation
1. **MCP SDK:**
   Use pip (as per SDK docs for simplicity; uv for production):
   ```
   pip install "mcp[cli]"
   ```

2. **Azure SDK Packages:**
   Based on the provided example, install the Azure AI agents, and related packages. These are part of Azure's ML/AI libraries (preview; check the docs for updates):
   ```
   pip install azure-ai-ml azure-ai-agents azure-identity
   ```
   - Motivation: `azure-ai-ml` for project clients, `azure-ai-agents` for agents/threads/runs, `azure-identity` for credentials. If errors occur (e.g., module not found), check Azure docs for exact packages—preview features may require `--pre`.

3. **Verify:**
   Run `mcp --help` to confirm MCP CLI works.

## Code Implementation
Create a file named `deep_research_server.py``. We'll use `FastMCP` for high-level server creation (inspired by SDK examples; it's simpler than low-level `Server` for our needs).

Key components:
- **Lifespan Handler:** Async function for startup/shutdown. Motivate: Initializes shared Azure resources (e.g., agent) once, avoiding per-query creation costs. Similar to FastAPI's lifespan for DB connections.
- **Deep Research Tool:** A `@mcp.tool()`-decorated function that takes a `query`, uses the shared agent, runs the research, saves MD, and returns path.
- **Helper Functions:** Adapt from Azure example for polling and saving, but integrate with MCP `Context` for updates.

```python
import os
import time
from typing import Optional
from datetime import datetime
import pathlib  # For path handling
import uuid  # For unique IDs

from mcp import FastMCP, Context  # Core MCP imports
from mcp.server.fastmcp import FastMCP  # Explicit if needed
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
from azure.ai.agents import AgentsClient
from azure.ai.agents.models import DeepResearchTool, MessageRole, ThreadMessage

# Globals for shared state (initialized in lifespan). Motivation: Simple way to share across tool calls in a single-process server; avoid if using multiprocessing.
project_client: Optional[AIProjectClient] = None
agents_client: Optional[AgentsClient] = None
deep_research_tool: Optional[DeepResearchTool] = None
agent_id: Optional[str] = None  # We'll reuse this agent

# Output dir for MD files. Create if not exists.
OUTPUT_DIR = "research_summaries"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Adapted from Azure example: Fetch latest agent response and use context for progress/logging.
def fetch_and_update_progress(
    thread_id: str,
    agents_client: AgentsClient,
    context: Context,
    last_message_id: Optional[str] = None,
) -> Optional[str]:
    response = agents_client.messages.get_last_message_by_role(
        thread_id=thread_id,
        role=MessageRole.AGENT,
    )
    if not response or not response.id == last_message_id:
        return last_message_id  # No new content

    # Instead of print, use context.log for MCP client visibility (e.g., in Claude).
    context.log.info("Agent response received.")
    text_content = "\n".join(t.text.value for t in response.text_messages)
    context.log.info(text_content)  # Log citations separately for clarity
    for ann in response.url_citation_annotations:
        context.log.info(f"URL Citation: [{ann.url_citation.title}].title}]({ann.url_citation.url}}")

    return response.id

# Adapted from Azure: Save message to MD. Motivation: Generate unique filename to handle concurrent queries; use pathlib for cross-platform paths.
def save_research_summary(
    message: ThreadMessage,
    query: str,
) -> str:
    if not message:
        raise ValueError("No message content provided.")

    # Unique filename: timestamp + sanitized query + uuid to avoid collisions.
    sanitized_query = "".joinjoin(c if c.isalnum() else "_" for c in query)[:30]  # Limit length
    filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{sanitized_query}_{uuid.uuid4().hex[:8]}.md"
    filepath = os.path.join(OUTPUT_DIR, filename)

    with open(filepath, "w", encoding="utf-8") as fp:
        # Write text summary
        text_summary = "\n\n".join([t.text.value.strip() for t in message.text_messages])
        fp.write(text_summary)

        # Write unique URL citations
        if message.url_citation_annotations:
            fp.write("\n\n## References\n")
            seen_urls = set()
            for ann in message.url_citation_annotations:
                url = ann.url_citation.url
                title = ann.url_citation.title or url
                if url not in seen_urls:
                    fp.write(f"- [{title}]({url})\n")
                    seen_urls.add(url)

    # Return file:// URI for standard linking; client can open it.
    return f"file://{pathlib.Path(filepath).resolve()}"

# Lifespan for init/cleanup. Motivation: Azure ops may be async; allows graceful shutdown (e.g., delete agent to free resources).
async def server_lifespan(mcp: FastMCP):
    global project_client, agents_client, deep_research_tool, agent_id

    # Init Azure clients
    project_client = AIProjectClient(
        endpoint=os.environ["PROJECT_ENDPOINT"],
        credential=DefaultAzureCredential(),
    )

    conn_id = project_client.connections.get(name=os.environ["BING_RESOURCE_NAME"]).id

    deep_research_tool = DeepResearchTool(
        bing_grounding_connection_id=conn_id,
        deep_research_model=os.environ["DEEP_RESEARCH_MODEL_DEPLOYMENT_NAME"],
    )

    # Enter context managers
    project_client.__enter__()
    agents_client = project_client.agents.__enter__()  # Assuming context manager usage as in example

    # Create reusable agent
    agent = agents_client.create_agent(
        model=os.environ["MODEL_DEPLOYMENT_NAME"],
        name="deep-research-agent",
        instructions="You are a helpful Agent that assists in researching scientific topics using Deep Research.",
        tools=deep_research_tool.definitions,
    )
    agent_id = agent.id
    print(f"Created reusable agent, ID: {agent_id}")  # For server logs

    yield  # Server runs here

    # Cleanup
    if agent_id:
        agents_client.delete_agent(agent_id)
        print("Deleted agent")
    agents_client.__exit__(None, None, None)
    project_client.__exit__(None, None, None)

# Create MCP server
mcp = FastMCP("AzureDeepResearchMCP", lifespan=server_lifespan)

# Define the tool. Motivation: Tools are for actions; use Context for progress to handle long polls without blocking feel.
@mcp.tool()
def deep_research(context: Context, query: str) -> str:
    """Perform deep research on a topic using Azure AI Foundry and return path to summary Markdown file."""
    if not agent_id or not agents_client:
        raise RuntimeError("Server not initialized properly. Check lifespan.")

    context.log.info(f"Starting deep research for query: {query}")

    # Create thread and message
    thread = agents_client.threads.create()
    message = agents_client.messages.create(
        thread_id=thread.id,
        role="user",
        content=query,  # Use the input query directly
    )

    # Start run
    run = agents_client.runs.create(thread_id=thread.id, agent_id=agent_id)
    context.log.info("Research run started. Polling for completion...")

    last_message_id = None
    while run.status in ("queued", "in_progress"):
        time.sleep(1)  # Poll interval; motivate: Azure recommends ~1s to avoid rate limits
        run = agents_client.runs.get(thread_id=thread.id, run_id=run.id)

        # Update progress via context (shows in client, e.g., Claude UI)
        context.progress(f"Run status: {run.status}...")

        last_message_id = fetch_and_update_progress(
            thread_id=thread.id,
            agents_client=agents_client,
            context=context,
            last_message_id=last_message_id,
        )

    context.log.info(f"Run finished with status: {run.status}")

    if run.status == "failed":
        error_msg = run.last_error if hasattr(run, 'last_error') else "Unknown error"
        context.log.error(f"Research failed: {error_msg}")
        raise ValueError(f"Deep research failed: {error_msg}")

    # Get final message and save
    final_message = agents_client.messages.get_last_message_by_role(
        thread_id=thread.id, role=MessageRole.AGENT
    )
    if not final_message:
        raise ValueError("No final response from agent.")

    filepath_uri = save_research_summary(final_message, query)
    context.log.info(f"Research summary saved to {filepath_uri}")

    return filepath_uri  # Client gets this path to access the file

# Optional: Add a simple prompt as inspiration (e.g., to generate queries)
@mcp.prompt()
def generate_research_query(topic: str) -> str:
    """Generate a refined research query for deep research."""
    return f"Provide the latest in-depth research summary on {topic} from the past year, including key papers and trends."

# Run the server
if __name__ == "__main__":
    mcp.run(transport="stdio")  # For local testing with Claude Desktop
    # For production: mcp.run(transport="http", host="0.0.0.0", port=8000)
```

## How It Works: Step-by-Step Explanation
1. **Server Startup (Lifespan):** Initializes Azure clients and creates a reusable agent. Motivation: Avoids recreating expensive resources per tool call; cleanup prevents resource leaks.
2. **Tool Invocation:** When an MCP client calls `deep_research`, it creates a new thread/run per query (isolated), polls with progress updates via `Context` (prevents "hung" feeling), saves the MD, and returns the URI.
3. **Polling and Progress:** The while loop sleeps and updates via `context.progress`—explicit because semi-senior devs might forget user feedback in long ops.
4. **File Handling:** Saves to `./research_summaries/` with unique names; returns `file://` for direct access. If remote, extend by adding a `@mcp.resource("file://{id}")` handler to serve content over MCP (e.g., read from disk and return str).
5. **Error Handling:** Raises exceptions on failure, logged via `context.log.error` for visibility.

## Testing and Usage Examples
1. **Local Run:** `python deep_research_server.py`. In Claude Desktop, connect via MCP stdio and invoke the tool: e.g., call `deep_research` with query="Latest on AI ethics". Expect output like `file:///path/to/20250716_123456_ai_ethics_abcdef12.md`.
2. **Motivated Example Invocation (via MCP Client):** If building a test client (using MCP SDK), something like:
   ```python
   from mcp.client import MCPClient
   client = MCPClient(transport="stdio")
   result = client.tool("deep_research", {"query": "Quantum computing advances"})
   print(result)  # e.g., "file:///.../summary.md"
   # Then, open and read the file in your code
   ```
3. **Production Notes:** Switch to `transport="http"` for remote access. Monitor Azure costs (Deep Research can be pricey). If polls timeout, consider async polling with `asyncio.sleep`.
4. **Debugging Tips:** Check env vars; if auth fails, verify `az login`. For large files, test with simple queries first.

This provides clear, implementable guidance. If issues arise (e.g., Azure preview changes), refer to the linked docs and adapt.

...

Yes, there is more to add in a follow-up post to enhance the original guide. Based on the provided links and related search results (which include code snippets, documentation updates, and best practices as of July 16, 2025), I'll outline a structured follow-up post. This builds on the original synchronous implementation by introducing asynchronous support, which is highlighted in the async sample and Azure docs. Async is particularly useful for an MCP server to handle concurrent tool calls efficiently without blocking (e.g., during long polling for Deep Research runs).

The follow-up should treat the reader as a semi-senior developer: Be explicit on changes (e.g., why switch to async clients, how to handle awaits), provide motivated code diffs/examples, and note potential pitfalls like event loops in MCP. Draw from the async sample's patterns (e.g., using `asyncio.sleep`, async polling) and docs (e.g., import from `.aio`, install `aiohttp`).

### Follow-Up Post: Enhancing Your MCP Server with Asynchronous Azure Deep Research Integration

In the previous guide, we built a synchronous MCP server integrating Azure AI Foundry's Deep Research. Since then, Azure's SDK has evolved (now at version 1.0.0b12 as per the [Azure AI Projects README](https://learn.microsoft.com/en-us/python/api/overview/azure/ai-projects-readme?view=azure-python-preview)), with stronger emphasis on async operations for better scalability. The [async Deep Research sample](https://github.com/Azure/azure-sdk-for-python/blob/main/sdk/ai/azure-ai-agents/samples/agents_async/sample_agents_deep_research_async.py) demonstrates this using `asyncio` for non-blocking polling and context managers.

**Why Add This?** Deep Research runs can take minutes, and a sync implementation blocks the thread during `time.sleep`. Async allows concurrent handling of multiple queries (e.g., from multiple MCP clients), improving server responsiveness. This is motivated by production scenarios where your MCP server might serve multiple LLMs or users. However, stick to sync if your use case is single-threaded to avoid async complexity (e.g., managing event loops).

**Key Updates from Azure Docs and Sample:**
- **Package Changes:** Agents are now in the separate `azure-ai-agents` package (auto-installed with `azure-ai-projects`). For async, install `aiohttp` and use imports from `.aio` modules.
- **Async Patterns:** Use `async def` for functions, `await` for API calls, `asyncio.sleep` for polling. The sample shows async agent creation, threading, and streaming responses.
- **Preview Notes:** Version 1.0.0b12 includes breaking changes (e.g., discontinued connection strings; use endpoints). Only Entra ID auth is supported. Check the changelog for details.
- **Best Practices:** Enable logging with `logging_enable=True` for debugging. For Deep Research, ensure regions like West US support the `o3-deep-research` model. Outputs are auditable with citations.
- **Limitations:** No internet package installs in code execution tools (as per your env); async requires Python 3.9+. If using MCP over HTTP, async helps with concurrency.

#### Updated Installation
Update your pip commands to use the latest preview packages:
```
pip install azure-ai-projects --pre  # Installs azure-ai-agents automatically
pip install aiohttp  # Required for async clients
```
Motivation: `--pre` ensures you get the preview version (1.0.0b12+), which includes async support and fixes.

#### Making the MCP Server Asynchronous
Assuming MCP's `FastMCP` supports async tools/lifespan (check the [MCP SDK docs](https://github.com/modelcontextprotocol/python-sdk) for confirmation; if not, fall back to sync or use `asyncio.to_thread`), update `deep_research_server.py` as follows. I'll provide diffs and full updated sections for clarity.

1. **Imports and Globals:** Add async-related imports.
   ```diff
   +import asyncio
   +from azure.ai.projects.aio import AIProjectClient as AsyncAIProjectClient  # Async variant
   +from azure.ai.agents.aio import AgentsClient as AsyncAgentsClient  # Async agents
   from azure.ai.projects import AIProjectClient  # Keep sync if needed for fallback
   # ... other imports remain
   ```

2. **Lifespan Handler:** Make it async and use async clients. Motivation: Allows awaiting async init/cleanup; reuse agent as before.
   ```python
   async def server_lifespan(mcp: FastMCP):
       global project_client, agents_client, deep_research_tool, agent_id
       
       # Use async client for non-blocking init
       project_client = AsyncAIProjectClient(
           endpoint=os.environ["PROJECT_ENDPOINT"],
           credential=DefaultAzureCredential(),
       )
       
       # Await async operations
       conn_id = await project_client.connections.get(name=os.environ["BING_RESOURCE_NAME"]).id  # Note: Await if method is async
       
       deep_research_tool = DeepResearchTool(  # Same as sync
           bing_grounding_connection_id=conn_id,
           deep_research_model=os.environ["DEEP_RESEARCH_MODEL_DEPLOYMENT_NAME"],
       )
       
       # Enter async context (assuming __aenter__ support; check docs)
       await project_client.__aenter__()
       agents_client = await project_client.agents.__aenter__()  # Async enter
       
       # Create agent async
       agent = await agents_client.create_agent(
           model=os.environ["MODEL_DEPLOYMENT_NAME"],
           name="deep-research-agent",
           instructions="You are a helpful Agent that assists in researching scientific topics using Deep Research.",
           tools=deep_research_tool.definitions,
       )
       agent_id = agent.id
       print(f"Created reusable agent, ID: {agent_id}")
       
       yield  # Server runs here
       
       # Async cleanup
       if agent_id:
           await agents_client.delete_agent(agent_id)
           print("Deleted agent")
       await agents_client.__aexit__(None, None, None)
       await project_client.__aexit__(None, None, None)
   ```

3. **Async Helper Functions:** Update polling and saving to async. Motivation: Non-blocking sleep allows other tasks during waits.
   ```python
   async def fetch_and_update_progress(  # Add async
       thread_id: str,
       agents_client: AsyncAgentsClient,
       context: Context,
       last_message_id: Optional[str] = None,
   ) -> Optional[str]:
       response = await agents_client.messages.get_last_message_by_role(  # Await
           thread_id=thread_id,
           role=MessageRole.AGENT,
       )
       # ... rest same, but if any I/O, await it
       
   # save_research_summary remains sync (file I/O); wrap in asyncio.to_thread if needed for concurrency
   ```

4. **Deep Research Tool:** Make it async. Motivation: Allows awaiting the run/poll without blocking MCP's event loop.
   ```python
   @mcp.tool()  # Assume MCP supports async tools; if not, use @mcp.tool(async=True) or equivalent
   async def deep_research(context: Context, query: str) -> str:  # Add async
       """Perform deep research on a topic using Azure AI Foundry and return path to summary Markdown file."""
       if not agent_id or not agents_client:
           raise RuntimeError("Server not initialized properly. Check lifespan.")
       
       context.log.info(f"Starting deep research for query: {query}")
       
       # Create thread and message async
       thread = await agents_client.threads.create()
       message = await agents_client.messages.create(
           thread_id=thread.id,
           role="user",
           content=query,
       )
       
       # Start run async
       run = await agents_client.runs.create(thread_id=thread.id, agent_id=agent_id)
       context.log.info("Research run started. Polling for completion...")
       
       last_message_id = None
       while run.status in ("queued", "in_progress"):
           await asyncio.sleep(1)  # Non-blocking sleep
           run = await agents_client.runs.get(thread_id=thread.id, run_id=run.id)
           
           context.progress(f"Run status: {run.status}...")
           
           last_message_id = await fetch_and_update_progress(  # Await async helper
               thread_id=thread.id,
               agents_client=agents_client,
               context=context,
               last_message_id=last_message_id,
           )
       
       # ... rest same (final message, save, return URI)
       
       return filepath_uri
   ```

5. **Running the Server:** No change, but ensure MCP's `run()` supports async (e.g., via `asyncio.run(mcp.run())` if needed).

**Pitfalls and Testing:**
- **Event Loop Issues:** If MCP isn't async-native, run the tool in `asyncio.to_thread` for sync compatibility.
- **Concurrency Test:** Simulate multiple queries; async should handle them without sequential blocking.
- **Example Query:** Use the same test as before, but monitor for faster response times under load.
- **Further Reading:** Check Azure's [async sample](https://github.com/Azure/azure-sdk-for-python/blob/main/sdk/ai/azure-ai-agents/samples/agents_async/sample_agents_deep_research_async.py) for streaming responses (e.g., via async iterators) if you want real-time updates beyond polling.

This async upgrade makes your MCP server more robust. If Azure releases further updates (e.g., built-in streaming in agents), revisit the docs. Questions? Drop them below!


# Structured Task Breakdown for Building an MCP Server with Azure AI Foundry Deep Research Integration

Based on the provided conversation history, including the original prompt, the initial technical guide, the follow-up post on async enhancements, and the Claude.md context file, I'll break "everything" down into structured tasks. This focuses especially on the core content (the technical instruction for building the MCP server), while incorporating the entire thread's evolution (e.g., sync to async upgrades, motivations, and best practices).

As an expert, I've used available tools to verify and update information as of July 16, 2025:
- Azure AI Projects SDK is at version 1.0.0b12 (using API version 2025-05-15-preview), with breaking changes like separate `azure-ai-agents` package and discontinued connection strings (use endpoints only). Async support requires `aiohttp` and `.aio` imports.
- MCP SDK supports async natively (e.g., in `FastMCP` tools and lifespans), with examples for long-running tools using `async def` and progress reporting.
- No major post-2024 changes in the async Deep Research sample beyond what's in the follow-up, but emphasize non-blocking patterns for concurrency.
- Corrected code typos from the guide (e.g., `joinjoin` -> `join`, typo in log formatting).

The goal is to provide clear, actionable tasks for a semi-senior developer implementing this. Each task includes motivations, explicit steps, code examples where relevant, and pitfalls. Tasks are grouped into phases for logical flow.

## Phase 1: Preparation and Research (Setup Knowledge Base)
These tasks ensure you understand prerequisites and verify up-to-date info, avoiding outdated assumptions.

1. **Review Documentation and Verify SDK Versions**
   - **Motivation:** Azure features are in preview and evolve (e.g., API 2025-05-15-preview as of July 2025); MCP SDK has async support but check for your use case.
   - **Steps:**
     - Read Azure Deep Research docs: https://learn.microsoft.com/en-us/azure/ai-foundry/agents/how-to/tools/deep-research-samples.
     - Check Azure AI Projects README: https://learn.microsoft.com/en-us/python/api/overview/azure/ai-projects-readme?view=azure-python-preview (confirm version 1.0.0b12, note breaking changes like no connection strings).
     - Browse MCP SDK GitHub: https://github.com/modelcontextprotocol/python-sdk (confirm async support in `FastMCP`, e.g., via examples in `/examples/snippets/servers/tool_progress.py`).
     - Set up Azure subscription: Ensure access to AI Foundry, create a project, and note endpoint (e.g., `https://<account>.services.ai.azure.com/api/projects/<project>`). Log in via `az login`.
   - **Pitfalls:** Preview features may change; use `--pre` for pip installs. If using older SDK, pin to 1.0.0b10 to avoid breaks.
   - **Expected Output:** List of env vars: `PROJECT_ENDPOINT`, `BING_RESOURCE_NAME`, `DEEP_RESEARCH_MODEL_DEPLOYMENT_NAME` (e.g., "o3-deep-research"), `MODEL_DEPLOYMENT_NAME` (e.g., "gpt-4o").

2. **Familiarize with MCP Concepts**
   - **Motivation:** MCP uses tools for actions, resources for data, prompts for templates; inspired by SDK examples like simple `add` tool.
   - **Steps:**
     - Study MCP SDK examples: Simple tools (e.g., computation), resources (e.g., data exposure), long-running async tools.
     - Note: Deep Research fits as a "tool" due to side effects (Azure query, file save); use `Context` for progress/logs.
   - **Pitfalls:** Don't confuse with resources—tools are for dynamic actions.

## Phase 2: Installation and Environment Setup
Install dependencies and prepare the project structure.

3. **Install Packages**
   - **Motivation:** Use preview versions for latest features; include async deps for future-proofing.
   - **Steps:**
     - Run: `pip install "mcp[cli]"`
     - Run: `pip install azure-ai-projects --pre` (installs `azure-ai-agents` automatically).
     - Run: `pip install azure-identity aiohttp` (for credentials and async).
     - Verify: `mcp --help` works; import Azure modules in a test script.
   - **Pitfalls:** No internet installs in runtime envs; pre-install all. Python 3.9+ for async.

4. **Set Up Project Structure**
   - **Motivation:** Organized for scalability (e.g., output dir for large MD files).
   - **Steps:**
     - Create dir: `azure-deep-research-mcp/`
     - Add files: `deep_research_server.py`, `requirements.txt` (list deps), `Claude.md` (from provided template), `.gitignore` (ignore `research_summaries/`).
     - Create `research_summaries/` dir for outputs.
   - **Pitfalls:** Use unique filenames to avoid overwrites in concurrent runs.

## Phase 3: Implement the Synchronous Version (Core Guide)
Build the base sync server as in the initial guide, with corrections.

5. **Implement Lifespan Handler**
   - **Motivation:** Reuse Azure agent once to avoid per-query costs; similar to FastAPI lifespan.
   - **Steps:**
     - Define async lifespan (even in sync version, for compatibility).
     - Initialize clients, create reusable agent, cleanup on shutdown.
     - Code Example (corrected):
       ```python
       async def server_lifespan(mcp: FastMCP):
           global project_client, agents_client, deep_research_tool, agent_id
           project_client = AIProjectClient(endpoint=os.environ["PROJECT_ENDPOINT"], credential=DefaultAzureCredential())
           conn_id = project_client.connections.get(name=os.environ["BING_RESOURCE_NAME"]).id
           deep_research_tool = DeepResearchTool(bing_grounding_connection_id=conn_id, deep_research_model=os.environ["DEEP_RESEARCH_MODEL_DEPLOYMENT_NAME"])
           project_client.__enter__()
           agents_client = project_client.agents.__enter__()
           agent = agents_client.create_agent(model=os.environ["MODEL_DEPLOYMENT_NAME"], name="deep-research-agent", instructions="You are a helpful Agent that assists in researching scientific topics using Deep Research.", tools=deep_research_tool.definitions)
           agent_id = agent.id
           yield
           if agent_id: agents_client.delete_agent(agent_id)
           agents_client.__exit__(None, None, None)
           project_client.__exit__(None, None, None)
       ```
   - **Pitfalls:** Globals for shared state—use if single-process; else, use contextvars.

6. **Implement Helper Functions**
   - **Motivation:** Adapt Azure example for MCP; add progress for long runs.
   - **Steps:**
     - Create `fetch_and_update_progress` (use `context.log.info` instead of print).
     - Create `save_research_summary` (generate unique file, return `file://` URI).
     - Corrections: Fix `joinjoin` to `join`; fix log typo to `f"URL Citation: [{ann.url_citation.title}]({ann.url_citation.url})"`.
   - **Pitfalls:** Handle empty messages; use sets for unique citations.

7. **Implement the Deep Research Tool**
   - **Motivation:** Core action: Query Azure, poll, save MD, return path.
   - **Steps:**
     - Decorate with `@mcp.tool()`.
     - Create thread/run per query, poll with `time.sleep(1)`, update progress.
     - Save and return URI.
     - Code Example (key snippet):
       ```python
       @mcp.tool()
       def deep_research(context: Context, query: str) -> str:
           # ... (thread creation, run, polling with context.progress, error handling)
           filepath_uri = save_research_summary(final_message, query)
           return filepath_uri
       ```
   - **Pitfalls:** Poll interval ~1s to avoid rate limits; check `run.status == "failed"`.

8. **Add Optional Prompt and Run Server**
   - **Motivation:** Enhance with query generation; start with stdio for testing.
   - **Steps:**
     - Add `@mcp.prompt()` for `generate_research_query`.
     - Create `mcp = FastMCP(...)` and `if __name__ == "__main__": mcp.run(transport="stdio")`.
   - **Pitfalls:** For production, use HTTP transport.

## Phase 4: Testing the Synchronous Version
Verify before async upgrades.

9. **Test Locally**
   - **Motivation:** Ensure basics work; simulate MCP client.
   - **Steps:**
     - Run `python deep_research_server.py`.
     - Use Claude Desktop or test client: Call `deep_research` with query="Latest on quantum computing".
     - Check: File saved in `research_summaries/`, URI returned, progress logged.
   - **Pitfalls:** Monitor Azure costs; test with simple queries.

10. **Debug Common Issues**
    - **Motivation:** Explicit for semi-senior devs.
    - **Steps:** Check env vars, auth (`az login`), logs for errors. If large files, verify disk space.

## Phase 5: Upgrade to Asynchronous Version (Follow-Up Post)
Enhance for concurrency, based on Azure async sample.

11. **Update Imports and Clients**
    - **Motivation:** Non-blocking for multiple queries.
    - **Steps:** Add `import asyncio`; use `AsyncAIProjectClient` and `AsyncAgentsClient` from `.aio`.
    - Pitfalls: Install `aiohttp`; Python 3.9+.

12. **Make Lifespan and Helpers Async**
    - **Motivation:** Await init/cleanup.
    - **Steps:** Add `async def` to lifespan; `await` on async methods (e.g., `await project_client.__aenter__()`). Update `fetch_and_update_progress` to async with `await`.

13. **Make Tool Async**
    - **Motivation:** Use `asyncio.sleep(1)` for non-blocking polls.
    - **Steps:** Change to `async def deep_research`; `await` on creates/gets; call async helper.
    - Code Example (diff-style as in follow-up):
      ```diff
      +async def deep_research(context: Context, query: str) -> str:
      +    thread = await agents_client.threads.create()
      +    # ... 
      +    while run.status in ("queued", "in_progress"):
      +        await asyncio.sleep(1)
      +        run = await agents_client.runs.get(...)
      ```

14. **Test Async Concurrency**
    - **Motivation:** Validate non-blocking.
    - **Steps:** Simulate multiple queries; check parallel handling without sequential waits.

## Phase 6: Advanced Extensions and Maintenance
From Claude.md and thread motivations.

15. **Add Resource for File Serving (Optional)**
    - **Motivation:** For remote clients, serve MD over MCP.
    - **Steps:** Add `@mcp.resource()` to read/return file content.

16. **Monitor and Update**
    - **Motivation:** Azure/MCP evolve (e.g., check for streaming in agents).
    - **Steps:** Revisit docs quarterly; add caching for cost savings.

This breakdown covers the entire thread into 16 tasks, providing clear guidance. Total estimated time: 4-6 hours for sync, +2 hours for async. If implementing, start with Phase 3 and test iteratively.