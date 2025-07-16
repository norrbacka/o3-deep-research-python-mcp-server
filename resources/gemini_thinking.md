

# **Building a Production-Grade MCP Server with Azure AI Foundry Deep Research Integration**

## **Section 1: Foundational Concepts and Environment Setup**

This section establishes the necessary background knowledge for building a robust Model Context Protocol (MCP) server. It details the core principles of MCP, the architecture of the Azure AI Foundry Deep Research service that the server will integrate with, and a meticulous guide to configuring the development environment. A clear understanding of these fundamentals is essential before proceeding to implementation, as it informs the architectural decisions made throughout the project.

### **1.1. The Model Context Protocol (MCP) and FastMCP: The "API Layer for AI"**

The Model Context Protocol (MCP) is an open standard designed to create a standardized, secure, and two-way communication channel between Large Language Models (LLMs) and external systems, such as data sources, APIs, and other tools.1 Often described as the "USB-C port for AI," MCP provides a uniform interface that decouples the logic of an AI application from the specific tools and data it consumes.2 This standardization allows any MCP-compliant client (like an LLM agent) to interact with any MCP-compliant server without needing custom integration code for each connection.3

The protocol is built around three core primitives, which can be intuitively understood through analogies to a traditional REST API 5:

* **Resources:** These are analogous to GET endpoints in a REST API. They expose data that an LLM can read and load into its context to inform its responses or actions. Resources are designed to be idempotent and should not have side effects. An example would be a resource at the URI users://123/profile that returns a user's profile data.6  
* **Tools:** These are analogous to POST or PUT endpoints. They provide functionality that an LLM can invoke to perform an action, execute code, or produce a side effect. Unlike resources, tools are expected to change state or interact with external systems. The deep\_research function in this report is a prime example of an MCP tool, as it triggers a complex, stateful process in Azure.5  
* **Prompts:** These are reusable templates that structure and guide interactions with an LLM. They help ensure consistency and reliability for specific tasks by providing a pre-defined format for queries and responses.5

While the MCP specification can be implemented directly, this report will use **FastMCP**, a high-level Python framework designed to simplify the development of MCP servers and clients. FastMCP has been incorporated into the official MCP Python SDK and is the canonical way to build servers.9 It abstracts away the boilerplate code associated with protocol handling, server setup, and error management, allowing developers to focus on the core logic of their tools and resources by using simple, "Pythonic" decorators.4 This report will specifically use the

FastMCP class from the mcp.server.fastmcp module.1

### **1.2. Architecture of Azure AI Foundry's Deep Research Service**

The server's primary function is to integrate with Azure AI Foundry's Deep Research capability. This is not a simple, single-shot API call but a sophisticated, multi-step agentic process designed for in-depth research. It is a "code-only" release, accessible via the Azure AI Python SDKs, and is currently in public preview.12 Understanding its internal workflow is critical to appreciating why the MCP tool must be designed to handle long-running, asynchronous operations.

The Deep Research service follows a three-step flow to process a query:

1. **Intent Clarification and Scoping:** When a query is submitted, it is first processed by a standard GPT model (e.g., gpt-4o). This model's role is to analyze the user's prompt, clarify any ambiguities, and precisely define the scope of the research task. This initial step ensures that the subsequent, more resource-intensive stages are focused and efficient.12  
2. **Web Grounding with Bing Search:** Once the task is scoped, the agent invokes the Grounding with Bing Search tool. This tool gathers a curated set of recent, high-quality, and authoritative web sources relevant to the query. This grounding step is crucial for ensuring the final output is based on up-to-date, auditable information, rather than relying solely on the model's internal knowledge, which can be outdated.12 Usage of this service can incur costs.12  
3. **Deep Analysis and Synthesis:** The core of the service is the specialized o3-deep-research model. This model takes the grounded web sources and performs a deep analysis. It reasons step-by-step through the information, synthesizes insights from multiple sources, identifies patterns, and composes a comprehensive, nuanced summary. The final output includes citations, making every insight traceable back to its source, which is a key feature for transparency and auditing.13

This multi-step, asynchronous architecture means that a single research query can take several minutes to complete. The MCP server must therefore be designed to initiate this process, poll for its status over time, and handle the eventual result gracefully. The implementation will rely on the azure-ai-projects and the underlying azure-ai-agents Python SDKs, which are currently in a beta release cycle (e.g., version 1.0.0b12).14 This preview status implies that APIs may evolve, and developers should consult the official Microsoft documentation for the latest updates.

### **1.3. Comprehensive Environment Configuration**

A correct and complete environment setup is the most critical first step to prevent common configuration errors and ensure a smooth development process. This section provides a meticulous, step-by-step guide.

Python Environment Management  
It is strongly recommended to use uv, a modern and fast Python package installer and resolver, to manage the project's virtual environment and dependencies. This aligns with best practices suggested by the MCP SDK documentation.1 An alternative using  
pip is also viable.

To set up the environment with uv:

1. Initialize a new project: uv init mcp-deep-research-server  
2. Change into the project directory: cd mcp-deep-research-server  
3. Create and activate the virtual environment: uv venv and source.venv/bin/activate (on macOS/Linux) or .venv\\Scripts\\activate (on Windows).

Package Installation  
The following packages must be installed. Each serves a specific purpose in the server's architecture.

* "mcp\[cli\]": This installs the core Model Context Protocol SDK, including the FastMCP framework and the command-line interface for running and inspecting servers.1  
* azure-ai-projects \--pre: This is the primary Azure SDK for interacting with AI Foundry projects. It automatically includes the azure-ai-agents package as a dependency. The \--pre flag is essential to install the latest preview version that contains the Deep Research capabilities.14  
* azure-identity: This package provides the DefaultAzureCredential class, which simplifies authentication to Azure services by automatically using available credentials from the environment or a logged-in CLI session.17  
* aiohttp: This is a required dependency for the *asynchronous* Azure clients that will be used in the production-grade version of the server. It is best to install it upfront.

Install these packages using uv:  
uv add "mcp\[cli\]" "azure-ai-projects \--pre" azure-identity aiohttp  
Azure Authentication  
The server will use DefaultAzureCredential for authentication.17 This mechanism provides a seamless authentication flow that works both in local development and when deployed to Azure. It attempts to authenticate using a chain of methods, including environment variables, managed identity, and a logged-in Azure CLI session.18 For local development, the simplest approach is to log in to the Azure CLI using  
az login with an account that has the necessary permissions for the AI Foundry project.

Environment Variable Configuration  
A common source of failure during development is an incorrect or incomplete environment variable setup. The following table centralizes all required variables, explains their purpose, and provides guidance on where to obtain their values from the Azure portal. This table should be used as a checklist to create a .env file in the project's root directory.

| Variable Name                           | Purpose                                                      | How to Obtain                                                | Example Value                                    |
| :-------------------------------------- | :----------------------------------------------------------- | :----------------------------------------------------------- | :----------------------------------------------- |
| PROJECT\_ENDPOINT                       | The unique endpoint for your Azure AI Foundry project, used to initialize the client. | In the Azure Portal, navigate to your AI Project and find this on the **Overview** page. | https://your-proj.westus2.inference.ml.azure.com |
| BING\_RESOURCE\_NAME                    | The name of the Bing Search connection resource within your project. This is required for the web grounding step. | In the Azure Portal, navigate to your AI Project, then to **Management Center** \-\> **Connected resources**. | my-bing-connection                               |
| DEEP\_RESEARCH\_MODEL\_DEPLOYMENT\_NAME | The deployment name for the core o3-deep-research model used for analysis. | In the Azure Portal, navigate to your AI Project, then to the **Models \+ endpoints** tab. | o3-deep-research                                 |
| MODEL\_DEPLOYMENT\_NAME                 | The deployment name for the base LLM (e.g., GPT-4o) used for the initial intent clarification step. | In the Azure Portal, navigate to your AI Project, then to the **Models \+ endpoints** tab. | gpt-4o                                           |

With the environment fully configured, the project is ready for the implementation phase.

---

## **Section 2: Core Implementation: A Synchronous Deep Research Server**

This section details the development of the initial version of the MCP server. This implementation will be synchronous, making it easier to understand the core logic of interacting with the Azure Deep Research service before introducing the complexities of asynchronous programming. This synchronous server will serve as a fully functional baseline and a solid foundation for the production-grade enhancements that follow.

### **2.1. Server Architecture and Design Principles**

The primary architectural challenge is to reconcile the procedural nature of an MCP tool call with the stateful, long-running nature of the Azure Deep Research process. From the perspective of an MCP client, invoking a tool is conceptually a single, request-response operation. However, the underlying Azure process involves multiple distinct steps: creating a communication thread, adding a message, initiating a run, and then polling for the run's status until completion.

To bridge this "impedance mismatch," the MCP server will be designed as a **stateful orchestrator**. The deep\_research tool function will encapsulate the entire lifecycle of an Azure agent run. It will manage the creation of the thread, the submission of the query, the polling loop, and the final result handling, all within a single MCP tool invocation. This design makes the tool ergonomic for the client, which simply calls the tool with a query and waits for a result, oblivious to the complex orchestration happening on the server.

The implementation will be guided by the following design principles:

* **Efficiency through Lifespan Management:** The Azure clients and the reusable agent will be initialized only once when the server starts up. This is a critical performance optimization that avoids the high latency of re-authentication and resource creation on every tool call. This will be achieved using an MCP lifespan handler.  
* **Handling Large Outputs via Files:** The Deep Research service can generate extensive summaries. To avoid potential issues with message payload size limits in the MCP protocol and to provide a persistent artifact, the final Markdown summary will be saved to a local file. The tool will return the path to this file.  
* **Ensuring Request Isolation with Unique Filenames:** To handle concurrent requests without data corruption or overwrites, each saved summary file will be given a unique name, generated using a combination of a timestamp, a sanitized version of the query, and a universally unique identifier (UUID).

### **2.2. Implementing the Lifespan Handler for Resource Management**

In a long-running server application, it is inefficient and costly to establish connections to external services on every incoming request. The lifespan handler is the standard pattern in FastMCP (and its underlying frameworks, FastAPI and Starlette) for managing resources that should persist for the entire lifetime of the application.1

The initialization of the AIProjectClient involves network handshakes and authentication with Azure, making it a high-latency operation. A naive implementation that creates this client inside the tool function would introduce significant overhead to every call. The lifespan handler, which is implemented as an asynccontextmanager, provides the ideal solution. The code before the yield statement is executed once on server startup, and the code within the finally block is executed once on server shutdown.

By creating the AIProjectClient, the AgentsClient, and the reusable Azure agent within this lifespan context, the expensive initialization cost is paid only once. Subsequent tool calls can then reuse these initialized objects, making them significantly faster and more efficient. This is a non-obvious but essential best practice for building production-ready services.

The following code implements the server\_lifespan handler. It initializes the necessary Azure clients and creates a single, reusable agent that will be stored in global variables for access by the tool function.

Python

\# deep\_research\_server.py

import os  
import time  
from typing import Optional, AsyncIterator  
from datetime import datetime  
import pathlib  
import uuid  
from contextlib import asynccontextmanager

from mcp import FastMCP, Context  
from azure.ai.projects import AIProjectClient  
from azure.identity import DefaultAzureCredential  
from azure.ai.agents import AgentsClient  
from azure.ai.agents.models import DeepResearchTool, MessageRole, ThreadMessage

\# Globals for shared state, initialized in the lifespan handler.  
\# This is a simple approach for a single-process server.  
project\_client: Optional\[AIProjectClient\] \= None  
agents\_client: Optional\[AgentsClient\] \= None  
agent\_id: Optional\[str\] \= None

\# Create an output directory for the research summaries.  
OUTPUT\_DIR \= "research\_summaries"  
os.makedirs(OUTPUT\_DIR, exist\_ok=True)

@asynccontextmanager  
async def server\_lifespan(mcp: FastMCP) \-\> AsyncIterator\[None\]:  
    """  
    Manages the lifecycle of shared Azure resources.  
    Initializes clients and the agent on startup and cleans them up on shutdown.  
    """  
    global project\_client, agents\_client, agent\_id

    print("Server starting up... Initializing Azure clients.")  
      
    \# 1\. Initialize the main project client using credentials from the environment.  
    project\_client \= AIProjectClient(  
        endpoint=os.environ,  
        credential=DefaultAzureCredential(),  
    )
    
    \# 2\. Retrieve the connection ID for the Bing Search resource.  
    conn\_id \= project\_client.connections.get(name=os.environ).id
    
    \# 3\. Define the Deep Research tool with the Bing connection and the research model.  
    deep\_research\_tool \= DeepResearchTool(  
        bing\_grounding\_connection\_id=conn\_id,  
        deep\_research\_model=os.environ,  
    )
    
    \# 4\. Use context managers to handle the client sessions.  
    project\_client.\_\_enter\_\_()  
    agents\_client \= project\_client.agents.\_\_enter\_\_()
    
    \# 5\. Create a single, reusable agent. This avoids the overhead of creating an agent per request.  
    agent \= agents\_client.create\_agent(  
        model=os.environ,  
        name="mcp-deep-research-agent",  
        instructions="You are a helpful Agent that assists in researching scientific topics using the Deep Research tool.",  
        tools=deep\_research\_tool.definitions,  
    )  
    agent\_id \= agent.id  
    print(f"Azure agent created and ready. Agent ID: {agent\_id}")
    
    yield  \# The server runs while the context is active.
    
    \# 6\. Cleanup phase: This code runs on server shutdown.  
    print("Server shutting down... Deleting Azure agent.")  
    if agent\_id:  
        agents\_client.delete\_agent(agent\_id)  
        print("Azure agent deleted.")  
      
    \# Exit the client context managers gracefully.  
    if agents\_client:  
        agents\_client.\_\_exit\_\_(None, None, None)  
    if project\_client:  
        project\_client.\_\_exit\_\_(None, None, None)  
    print("Azure clients closed. Shutdown complete.")

\# Pass the lifespan handler to the FastMCP constructor.  
mcp \= FastMCP("AzureDeepResearchMCP", lifespan=server\_lifespan)

### **2.3. Crafting the deep\_research Tool**

With the server's resource management handled, the next step is to implement the core logic within the deep\_research tool. This function will orchestrate the interaction with the Azure agent service. A key aspect of this implementation is providing real-time feedback to the client during the long-running polling process.

A silent, blocking tool call that takes several minutes provides a poor user experience, as the client may assume the server has hung or crashed. The FastMCP Context object, which is automatically injected into any tool function that includes it as a type-hinted parameter, provides the solution.6 The

context object has logging and progress reporting methods (ctx.log.info(), ctx.progress()) that can be used to send real-time status updates back to the MCP client.

Inside the polling loop (while run.status in ("queued", "in\_progress")), a call to context.progress() will stream the current status of the Azure run. This transforms the operation from a frustrating black box into a transparent, long-running task, significantly improving the user experience.

The following code defines helper functions adapted from the Azure sample and the main deep\_research tool itself.

Python

\# deep\_research\_server.py (continued)

def fetch\_and\_print\_new\_agent\_response(  
    thread\_id: str,  
    agents\_client: AgentsClient,  
    last\_message\_id: Optional\[str\] \= None,  
) \-\> Optional\[str\]:  
    """  
    Fetches the latest agent response from the thread.  
    This is a helper adapted from the Azure example.  
    """  
    response \= agents\_client.messages.get\_last\_message\_by\_role(  
        thread\_id=thread\_id,  
        role=MessageRole.AGENT,  
    )  
    if not response or response.id \== last\_message\_id:  
        return last\_message\_id  \# No new content yet.

    print("\\n--- Agent Progress Update \---")  
    print("\\n".join(t.text.value for t in response.text\_messages))  
    for ann in response.url\_citation\_annotations:  
        print(f"URL Citation: \[{ann.url\_citation.title}\]({ann.url\_citation.url})")  
    print("---------------------------\\n")  
    return response.id

def save\_research\_summary(message: ThreadMessage, query: str) \-\> str:  
    """  
    Saves the final research summary to a unique Markdown file.  
    Returns the file URI of the saved document.  
    """  
    if not message:  
        raise ValueError("Cannot create research summary: no message content provided.")

    \# Generate a unique filename to prevent overwrites and collisions.  
    timestamp \= datetime.now().strftime('%Y%m%d\_%H%M%S')  
    sanitized\_query \= "".join(c for c in query if c.isalnum() or c in " \_-").rstrip()\[:30\]  
    unique\_id \= uuid.uuid4().hex\[:8\]  
    filename \= f"{timestamp}\_{sanitized\_query}\_{unique\_id}.md"  
      
    \# Use pathlib for robust, cross-platform path construction.  
    filepath \= pathlib.Path(OUTPUT\_DIR) / filename
    
    with open(filepath, "w", encoding="utf-8") as fp:  
        text\_summary \= "\\n\\n".join(\[t.text.value.strip() for t in message.text\_messages\])  
        fp.write(text\_summary)
    
        if message.url\_citation\_annotations:  
            fp.write("\\n\\n\#\# References\\n")  
            seen\_urls \= set()  
            for ann in message.url\_citation\_annotations:  
                url \= ann.url\_citation.url  
                title \= ann.url\_citation.title or url  
                if url not in seen\_urls:  
                    fp.write(f"- \[{title}\]({url})\\n")  
                    seen\_urls.add(url)  
      
    \# Return a file URI, which is a standard way to reference local files.  
    return filepath.resolve().as\_uri()

@mcp.tool()  
def deep\_research(context: Context, query: str) \-\> str:  
    """  
    Performs in-depth research on a given topic using Azure AI Foundry,  
    saves the result to a Markdown file, and returns the file path.  
      

    :param context: The MCP Context object, injected by FastMCP.  
    :param query: The research query string.  
    :return: A file URI pointing to the generated Markdown summary.  
    """  
    if not agent\_id or not agents\_client:  
        raise RuntimeError("Server is not properly initialized. Check the lifespan handler.")
    
    context.log.info(f"Received deep research request for query: '{query}'")
    
    \# 1\. Create a new, isolated thread for this specific research task.  
    thread \= agents\_client.threads.create()  
    context.log.info(f"Created Azure agent thread: {thread.id}")
    
    \# 2\. Create the initial user message in the thread.  
    agents\_client.messages.create(  
        thread\_id=thread.id,  
        role="user",  
        content=query,  
    )
    
    \# 3\. Start the research run. This is a non-blocking call that initiates the process.  
    run \= agents\_client.runs.create(thread\_id=thread.id, agent\_id=agent\_id)  
    context.log.info(f"Started Azure agent run: {run.id}. Polling for completion...")
    
    \# 4\. Poll for the run's status, providing progress updates to the client.  
    last\_message\_id \= None  
    while run.status in ("queued", "in\_progress"):  
        time.sleep(2)  \# Use a reasonable polling interval to avoid spamming the API.  
        run \= agents\_client.runs.get(thread\_id=thread.id, run\_id=run.id)  
          
        \# Send a progress update back to the MCP client.  
        context.progress(f"Research in progress... Current status: {run.status}")
    
        \# Optionally, print intermediate agent responses to the server console.  
        last\_message\_id \= fetch\_and\_print\_new\_agent\_response(  
            thread\_id=thread.id,  
            agents\_client=agents\_client,  
            last\_message\_id=last\_message\_id,  
        )
    
    context.log.info(f"Run finished with status: {run.status}")  
      
    \# Handle the final state of the run in the next section.  
    \#...

### **2.4. Managing Outputs and Failures**

Once the polling loop concludes, the server must handle the final state of the run. This involves checking for success or failure, processing the final message, and returning the appropriate response to the MCP client.

Robust error handling is a hallmark of a well-built service. The code must explicitly check if the run status is "failed". If it is, the server should extract the error details from the run.last\_error attribute, log this information for debugging purposes using context.log.error(), and then raise a Python exception. Raising an exception is the standard mechanism to signal a tool execution failure to the MCP client.

If the run is successful, the server fetches the final message from the agent, passes it to the save\_research\_summary helper function to create the Markdown file, and returns the resulting file:// URI to the client.

The following code completes the deep\_research tool function.

Python

\# deep\_research\_server.py (completing the deep\_research tool)

@mcp.tool()  
def deep\_research(context: Context, query: str) \-\> str:  
    """  
    Performs in-depth research on a given topic using Azure AI Foundry,  
    saves the result to a Markdown file, and returns the file path.  
      

    :param context: The MCP Context object, injected by FastMCP.  
    :param query: The research query string.  
    :return: A file URI pointing to the generated Markdown summary.  
    """  
    if not agent\_id or not agents\_client:  
        raise RuntimeError("Server is not properly initialized. Check the lifespan handler.")
    
    context.log.info(f"Received deep research request for query: '{query}'")
    
    thread \= agents\_client.threads.create()  
    context.log.info(f"Created Azure agent thread: {thread.id}")
    
    agents\_client.messages.create(  
        thread\_id=thread.id,  
        role="user",  
        content=query,  
    )
    
    run \= agents\_client.runs.create(thread\_id=thread.id, agent\_id=agent\_id)  
    context.log.info(f"Started Azure agent run: {run.id}. Polling for completion...")
    
    last\_message\_id \= None  
    while run.status in ("queued", "in\_progress"):  
        time.sleep(2)  
        run \= agents\_client.runs.get(thread\_id=thread.id, run\_id=run.id)  
        context.progress(f"Research in progress... Current status: {run.status}")  
        last\_message\_id \= fetch\_and\_print\_new\_agent\_response(  
            thread\_id=thread.id,  
            agents\_client=agents\_client,  
            last\_message\_id=last\_message\_id,  
        )
    
    context.log.info(f"Run finished with status: {run.status}")
    
    \# 5\. Handle failed runs.  
    if run.status \== "failed":  
        error\_details \= run.last\_error if hasattr(run, 'last\_error') and run.last\_error else "Unknown error"  
        error\_message \= f"Azure Deep Research run failed: {error\_details}"  
        context.log.error(error\_message)  
        raise ValueError(error\_message)
    
    \# 6\. On success, fetch the final message and save it to a file.  
    final\_message \= agents\_client.messages.get\_last\_message\_by\_role(  
        thread\_id=thread.id, role=MessageRole.AGENT  
    )  
    if not final\_message:  
        raise ValueError("Research run completed, but no final response was found from the agent.")
    
    filepath\_uri \= save\_research\_summary(final\_message, query)  
    context.log.info(f"Research summary successfully saved to: {filepath\_uri}")
    
    \# 7\. Return the file URI to the MCP client.  
    return filepath\_uri

### **2.5. Running and Testing the Synchronous Server**

With the complete synchronous implementation in deep\_research\_server.py, the server can be started and tested.

To run the server for local development and testing with a client like Claude Desktop, use the stdio transport. This transport communicates over the standard input and output streams of the process.

Execute the following command in the terminal:  
mcp run deep\_research\_server.py \--transport stdio  
Alternatively, to use the interactive MCP Inspector tool for debugging, run:  
mcp dev deep\_research\_server.py  
The MCP Inspector provides a web interface where tools can be invoked manually, and logs and progress updates can be viewed in real-time.1

When the tool is invoked with a query (e.g., "Give me the latest research into quantum computing over the last year."), the server console will show the startup logs, the progress updates from the polling loop, and finally, the log message indicating the file has been saved. The MCP client will receive a string response containing the file:// URI, such as file:///path/to/project/research\_summaries/20250716\_143000\_quantum\_computing\_abcdef12.md. The client can then access this local file to read the research summary.

---

## **Section 3: Advancing to a Production-Grade Asynchronous Server**

The synchronous server provides a functional and understandable baseline. However, for a production environment that may need to handle multiple concurrent requests, a synchronous architecture presents a significant bottleneck. This section details the process of refactoring the server to be fully asynchronous, transforming it into a high-throughput, responsive service.

### **3.1. The Imperative for Asynchronicity in I/O-Bound Services**

The deep\_research tool is a classic example of an **I/O-bound** operation. The majority of its execution time is not spent on CPU-intensive calculations but on waiting for network responses from the Azure services. In the synchronous implementation, the time.sleep(2) call blocks the entire server process. While the server is sleeping for one user's request, it is completely unable to accept or process new requests from other users. This sequential processing model severely limits the server's scalability and throughput.

Modern web frameworks and services are designed to solve this problem using asynchronous programming. Both the FastMCP framework (which is built on the asynchronous ASGI standard) and the Azure Python SDK are designed with asynchronicity as a core principle.1 The Azure SDK provides a parallel set of asynchronous clients in

.aio submodules (e.g., azure.ai.projects.aio.AIProjectClient) that are designed to be used with async and await.27

By leveraging these capabilities, the server can be converted to use await asyncio.sleep() instead of time.sleep(). When a task awaits an I/O operation, it yields control back to the server's event loop. The event loop is then free to work on other tasks, such as handling new incoming requests or progressing other polling loops that are ready. When the awaited I/O operation completes, the event loop resumes the original task. This cooperative multitasking model allows a single-process server to handle many concurrent requests efficiently, dramatically increasing its responsiveness and overall throughput. This is the single most important architectural upgrade for moving the service toward production readiness.

### **3.2. Refactoring the Implementation for Full Asynchronicity**

The process of converting the server from synchronous to asynchronous involves a series of targeted changes to the code. The core logic of the application remains the same, but the syntax is updated to use async/await, and the corresponding asynchronous library clients are used.

The following table provides a side-by-side comparison of the key changes required, making the refactoring process a clear and mechanical task. This visual aid helps demystify the conversion by highlighting the specific code swaps needed.

| Feature         | Synchronous Implementation                                   | Asynchronous Implementation                                  | Rationale for Change                                         |
| :-------------- | :----------------------------------------------------------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
| **Imports**     | from azure.ai.projects import AIProjectClient from azure.ai.agents import AgentsClient import time | from azure.ai.projects.aio import AIProjectClient from azure.ai.agents.aio import AgentsClient import asyncio | Switch to the async-native clients from the .aio submodules and use the asyncio library for non-blocking operations. |
| **Client Init** | project\_client \= AIProjectClient(...)                      | project\_client \= AIProjectClient(...)                      | The constructor signature is the same, but the imported class is the asynchronous version. |
| **Lifespan**    | project\_client.\_\_enter\_\_() agents\_client.delete\_agent(id) | await project\_client.\_\_aenter\_\_() await agents\_client.delete\_agent(id) | Use async context managers (\_\_aenter\_\_/\_\_aexit\_\_) and await all client methods that perform network I/O. |
| **Tool Def**    | @mcp.tool() def deep\_research(...)                          | @mcp.tool() async def deep\_research(...)                    | The @mcp.tool() decorator natively supports async def functions, enabling asynchronous tool implementations. |
| **API Calls**   | run \= agents\_client.runs.create(...)                       | run \= await agents\_client.runs.create(...)                 | Every method call that results in a network request to Azure must be awaited to be non-blocking. |
| **Polling**     | time.sleep(2)                                                | await asyncio.sleep(2)                                       | Use asyncio.sleep() to pause the current task without blocking the entire server's event loop. |

The following is the complete, final, and fully asynchronous version of deep\_research\_server.py. It incorporates all the changes outlined above and represents a production-grade implementation.

Python

\# deep\_research\_server\_async.py

import os  
import asyncio  
from typing import Optional, AsyncIterator  
from datetime import datetime  
import pathlib  
import uuid  
from contextlib import asynccontextmanager

from mcp import FastMCP, Context  
\# Import the asynchronous clients from the.aio submodules  
from azure.ai.projects.aio import AIProjectClient  
from azure.identity.aio import DefaultAzureCredential  
from azure.ai.agents.aio import AgentsClient  
from azure.ai.agents.models import DeepResearchTool, MessageRole, ThreadMessage

\# Globals for shared state remain the same  
project\_client: Optional\[AIProjectClient\] \= None  
agents\_client: Optional\[AgentsClient\] \= None  
agent\_id: Optional\[str\] \= None

OUTPUT\_DIR \= "research\_summaries"  
os.makedirs(OUTPUT\_DIR, exist\_ok=True)

@asynccontextmanager  
async def server\_lifespan(mcp: FastMCP) \-\> AsyncIterator\[None\]:  
    """Asynchronous lifespan handler for managing Azure resources."""  
    global project\_client, agents\_client, agent\_id

    print("Async server starting up... Initializing Azure clients.")  
      
    \# Use the async AIProjectClient and DefaultAzureCredential  
    project\_client \= AIProjectClient(  
        endpoint=os.environ,  
        credential=DefaultAzureCredential(),  
    )
    
    \# Await the async method call to get the connection ID  
    connection \= await project\_client.connections.get(name=os.environ)  
    conn\_id \= connection.id
    
    deep\_research\_tool \= DeepResearchTool(  
        bing\_grounding\_connection\_id=conn\_id,  
        deep\_research\_model=os.environ,  
    )
    
    \# Use async context managers for the clients  
    await project\_client.\_\_aenter\_\_()  
    agents\_client \= await project\_client.agents.\_\_aenter\_\_()
    
    \# Await the agent creation  
    agent \= await agents\_client.create\_agent(  
        model=os.environ,  
        name="mcp-deep-research-agent-async",  
        instructions="You are a helpful Agent that assists in researching scientific topics using the Deep Research tool.",  
        tools=deep\_research\_tool.definitions,  
    )  
    agent\_id \= agent.id  
    print(f"Azure agent created and ready. Agent ID: {agent\_id}")
    
    yield
    
    print("Async server shutting down... Deleting Azure agent.")  
    if agent\_id and agents\_client:  
        await agents\_client.delete\_agent(agent\_id)  
        print("Azure agent deleted.")  
      
    if agents\_client:  
        await agents\_client.\_\_aexit\_\_(None, None, None)  
    if project\_client:  
        await project\_client.\_\_aexit\_\_(None, None, None)  
    print("Azure clients closed. Shutdown complete.")

mcp \= FastMCP("AzureDeepResearchMCPAsync", lifespan=server\_lifespan)

\# The helper functions fetch\_and\_print... and save\_research\_summary remain synchronous  
\# as they perform console I/O and blocking file I/O, respectively. If file I/O  
\# became a bottleneck, it could be wrapped with asyncio.to\_thread.  
\# (Code for these helpers is omitted for brevity but is the same as the sync version).

@mcp.tool()  
async def deep\_research(context: Context, query: str) \-\> str:  
    """  
    Performs in-depth research asynchronously.  
    """  
    if not agent\_id or not agents\_client:  
        raise RuntimeError("Server is not properly initialized. Check the lifespan handler.")

    context.log.info(f"Received async deep research request for query: '{query}'")
    
    \# Await all async client operations  
    thread \= await agents\_client.threads.create()  
    context.log.info(f"Created Azure agent thread: {thread.id}")
    
    await agents\_client.messages.create(  
        thread\_id=thread.id,  
        role="user",  
        content=query,  
    )
    
    run \= await agents\_client.runs.create(thread\_id=thread.id, agent\_id=agent\_id)  
    context.log.info(f"Started Azure agent run: {run.id}. Polling for completion...")
    
    while run.status in ("queued", "in\_progress"):  
        \# Use asyncio.sleep for non-blocking waits  
        await asyncio.sleep(2)  
        run \= await agents\_client.runs.get(thread\_id=thread.id, run\_id=run.id)  
        context.progress(f"Research in progress... Current status: {run.status}")  
        \# (Optional printing to console remains synchronous)
    
    context.log.info(f"Run finished with status: {run.status}")
    
    if run.status \== "failed":  
        error\_details \= run.last\_error if hasattr(run, 'last\_error') and run.last\_error else "Unknown error"  
        error\_message \= f"Azure Deep Research run failed: {error\_details}"  
        context.log.error(error\_message)  
        raise ValueError(error\_message)
    
    final\_message \= await agents\_client.messages.get\_last\_message\_by\_role(  
        thread\_id=thread.id, role=MessageRole.AGENT  
    )  
    if not final\_message:  
        raise ValueError("Research run completed, but no final response was found from the agent.")
    
    \# File saving is a blocking operation, but typically fast enough not to warrant  
    \# wrapping in asyncio.to\_thread unless dealing with a very slow filesystem.  
    filepath\_uri \= save\_research\_summary(final\_message, query)  
    context.log.info(f"Research summary successfully saved to: {filepath\_uri}")
    
    return filepath\_uri

if \_\_name\_\_ \== "\_\_main\_\_":  
    \# The mcp.run() command handles the async event loop automatically.  
    mcp.run(transport="stdio")

### **3.3. Concurrency Considerations and Potential Pitfalls**

The asynchronous implementation significantly improves the server's ability to handle concurrent requests. However, there are still important considerations for a production deployment:

* **Shared State and Global Variables:** The use of global variables for the shared agents\_client and agent\_id is safe within a single-process asynchronous model. The event loop ensures that operations on these variables are not interleaved in a dangerous way. However, if this server were to be deployed using a multi-process worker model (e.g., with a process manager like Gunicorn), each worker process would have its own separate set of global variables and its own Azure agent. This is generally acceptable, but if a truly shared state across processes were required (e.g., for a shared cache of results), a different mechanism like an external Redis cache would be necessary.  
* **API Rate Limiting:** While the asyncio.sleep(2) call provides a basic, static delay to prevent overwhelming the Azure APIs, a more sophisticated production system might implement an exponential backoff strategy for polling. This would involve increasing the sleep duration after repeated checks if the status has not changed, reducing the overall number of API calls for very long-running research tasks.  
* **Resource Cleanup:** The lifespan handler ensures that the created Azure agent is deleted upon graceful server shutdown. If the server crashes or is terminated forcefully, the agent resource might be orphaned in Azure. Production environments should have monitoring and cleanup scripts to detect and remove such orphaned resources to prevent unnecessary costs and resource clutter.

---

## **Section 4: Deployment, Extensions, and Final Considerations**

With a robust, asynchronous server implementation complete, the final step is to consider its deployment, potential architectural extensions for greater flexibility, and the operational aspects of running it in a production environment. This section provides forward-looking advice to transform the project from a working script into a truly deployable and maintainable service.

### **4.1. Preparing for Deployment: Beyond Localhost**

The server has so far been run using the stdio transport, which is ideal for local development and integration with tools like Claude Desktop. However, for deployment as a remote service that can be accessed over a network, the transport protocol must be changed.

* **HTTP Transport:** FastMCP supports an HTTP-based transport that is recommended for web deployments.7 To run the server with this transport, the  
  mcp.run() command is modified:  
  mcp.run(transport="http", host="0.0.0.0", port=8000)  
  This will start the server and make it accessible on port 8000 of the host machine. This is the standard approach for deploying the server within a container (e.g., Docker) or on a virtual machine.  
* **Security:** When exposing the server over HTTP, security becomes a critical concern. The stdio transport is implicitly secure as it's only accessible locally, but an HTTP endpoint is open to the network. FastMCP has built-in support for configurable authentication providers to protect server endpoints.4 While a detailed implementation of authentication is beyond the scope of this guide, it is an essential next step for any production deployment. The developer should investigate  
  FastMCP's authentication documentation to secure the endpoint using industry-standard protocols.

### **4.2. Advanced Extension: Decoupling with an MCP Resource**

The current design, which returns a file:// URI, has a significant limitation: it creates a tight coupling between the client and the server's filesystem. This contract is brittle and will break if the server is deployed remotely. A client running on a user's machine has no access to a file:// path on a remote server running in an Azure Container App. The URI would be useless.

A more robust and portable architecture can be achieved by leveraging two MCP primitives in concert: a **tool** and a **resource**. This approach decouples the action of performing the research from the mechanism of retrieving the data, creating a more flexible and resilient service.

The improved architecture would work as follows:

1. The deep\_research **tool** would perform the research and save the file to disk as before. However, instead of returning a file:// URI, it would generate a unique ID for the run and return a logical, custom URI like research-summary://\<unique-run-id\>.  
2. A *new* @mcp.resource("research-summary://{run\_id}") handler would be created. The job of this resource handler is to take the run\_id from the URI, find the corresponding summary file on the server's disk, read its contents, and return the raw Markdown text as the resource's content.

This design has several advantages:

* **Decoupling:** The client no longer needs to know anything about the server's filesystem. It simply receives a logical URI and uses the standard MCP mechanism (read\_resource) to fetch its content.  
* **Portability:** The architecture works seamlessly over any transport (stdio or http), making the server fully portable without any code changes.  
* **Protocol-Native:** It uses the MCP primitives as intended—a tool for actions and a resource for data retrieval—leading to a cleaner and more idiomatic design.

The following code provides a skeleton for this advanced pattern:

Python

\# \--- In deep\_research\_server\_async.py \---

\# Modified deep\_research tool  
@mcp.tool()  
async def deep\_research(context: Context, query: str) \-\> str:  
    \#... (perform research and save file with a name based on a unique\_id)  
    unique\_id \= uuid.uuid4().hex  
    \#... save\_research\_summary\_with\_id(final\_message, query, unique\_id)  
      

    \# Return a logical URI instead of a file URI  
    return f"research-summary://{unique\_id}"

\# New resource handler  
@mcp.resource("research-summary://{run\_id}")  
def get\_research\_summary(run\_id: str) \-\> str:  
    """  
    Finds the research summary file by its run ID and returns its content.  
    """  
    \# Logic to find the file in OUTPUT\_DIR based on the run\_id in its filename.  
    \# For example, search for a file matching "\*\_{run\_id}.md".  
    try:  
        filepath \= next(pathlib.Path(OUTPUT\_DIR).glob(f"\*\_{run\_id}.md"))  
        return filepath.read\_text(encoding="utf-8")  
    except StopIteration:  
        \# In a real implementation, this should return a proper MCP error.  
        raise FileNotFoundError(f"No research summary found for run ID: {run\_id}")

### **4.3. Operational Excellence: Monitoring, Auditing, and Cost Management**

Deploying a service to production involves more than just writing the code. It requires ongoing operational management.

* **Monitoring:** The performance, usage, and potential errors of the deployed Azure models (both the GPT clarification model and the o3-deep-research model) should be actively monitored through the Azure portal. This helps in identifying performance bottlenecks, tracking usage patterns, and debugging issues.  
* **Auditing and Transparency:** A key strength of the Deep Research service is its ability to provide citations for the information it synthesizes.13 The process of saving the output to a Markdown file preserves this crucial audit trail. For enterprise applications where the provenance of information is important, this feature is invaluable.  
* **Cost Management:** It is critical to be aware that the services used in this implementation are not free. Both the underlying Azure AI models and the Grounding with Bing Search API calls will incur costs based on usage.12 For a production deployment, it is essential to understand the Azure pricing models for these services, set up budget alerts in the Azure portal, and implement controls if necessary to prevent unexpected expenses.

## **Conclusion**

This report has detailed the end-to-end process of designing, implementing, and preparing for the deployment of a sophisticated MCP server. The journey began with establishing a solid foundation in the core concepts of the Model Context Protocol and the architecture of Azure's AI Foundry Deep Research service. It then progressed through the development of a functional synchronous server, which served as a stepping stone to a robust, production-grade asynchronous implementation capable of handling high-concurrency workloads.

The final architecture demonstrates how to effectively orchestrate a complex, long-running external process within the MCP framework, providing a seamless experience for the client. Key practices, such as using lifespan handlers for efficient resource management, leveraging the Context object for real-time user feedback, and designing for extensibility by decoupling actions from data retrieval, have been highlighted.

By combining a standardized, open protocol like MCP with a powerful, specialized backend service like Azure AI Deep Research, developers can create high-value, reusable, and scalable components for modern AI application stacks. The resulting server is not merely a script but a well-architected service ready for integration into a broader enterprise ecosystem, capable of delivering auditable, in-depth research as a programmable and composable service.

#### **Works cited**

1. modelcontextprotocol/python-sdk: The official Python SDK ... \- GitHub, accessed on July 16, 2025, [https://github.com/modelcontextprotocol/python-sdk](https://github.com/modelcontextprotocol/python-sdk)  
2. Model Context Protocol (MCP) \- Anthropic API, accessed on July 16, 2025, [https://docs.anthropic.com/en/docs/mcp](https://docs.anthropic.com/en/docs/mcp)  
3. Demystifying the Model Context Protocol (MCP) with Python: A Beginner's Guide, accessed on July 16, 2025, [https://mostafawael.medium.com/demystifying-the-model-context-protocol-mcp-with-python-a-beginners-guide-0b8cb3fa8ced](https://mostafawael.medium.com/demystifying-the-model-context-protocol-mcp-with-python-a-beginners-guide-0b8cb3fa8ced)  
4. Welcome to FastMCP 2.0\! \- FastMCP, accessed on July 16, 2025, [https://gofastmcp.com/](https://gofastmcp.com/)  
5. fastmcp \- PyPI, accessed on July 16, 2025, [https://pypi.org/project/fastmcp/2.2.4/](https://pypi.org/project/fastmcp/2.2.4/)  
6. AI-App/ModelContextProtocol.Python-SDK: The official Python SDK for Model Context Protocol servers and clients \- GitHub, accessed on July 16, 2025, [https://github.com/AI-App/ModelContextProtocol.Python-SDK](https://github.com/AI-App/ModelContextProtocol.Python-SDK)  
7. A Beginner's Guide to Use FastMCP \- Apidog, accessed on July 16, 2025, [https://apidog.com/blog/fastmcp/](https://apidog.com/blog/fastmcp/)  
8. The Model Context Protocol (MCP) is a standardized way to supply context to large language models (LLMs). Using the MCP Python SDK, you can build servers that expose data (resources), functionality (tools), and interaction templates (prompts) to LLM applications in a secure and modular fashion. In this tutorial, we'll build a simple MCP server in P \- GitHub, accessed on July 16, 2025, [https://github.com/ruslanmv/Simple-MCP-Server-with-Python](https://github.com/ruslanmv/Simple-MCP-Server-with-Python)  
9. jlowin/fastmcp: The fast, Pythonic way to build MCP servers and clients \- GitHub, accessed on July 16, 2025, [https://github.com/jlowin/fastmcp](https://github.com/jlowin/fastmcp)  
10. FastMCP() vs. Server() with Python SDK? : r/mcp \- Reddit, accessed on July 16, 2025, [https://www.reddit.com/r/mcp/comments/1i282ii/fastmcp\_vs\_server\_with\_python\_sdk/](https://www.reddit.com/r/mcp/comments/1i282ii/fastmcp_vs_server_with_python_sdk/)  
11. fastmcp \- PyPI, accessed on July 16, 2025, [https://pypi.org/project/fastmcp/1.0/](https://pypi.org/project/fastmcp/1.0/)  
12. Deep research tool \- Azure AI Foundry | Microsoft Learn, accessed on July 16, 2025, [https://learn.microsoft.com/en-us/azure/ai-foundry/agents/how-to/tools/deep-research](https://learn.microsoft.com/en-us/azure/ai-foundry/agents/how-to/tools/deep-research)  
13. Introducing Deep Research in Azure AI Foundry Agent Service | Microsoft Azure Blog, accessed on July 16, 2025, [https://azure.microsoft.com/en-us/blog/introducing-deep-research-in-azure-ai-foundry-agent-service/](https://azure.microsoft.com/en-us/blog/introducing-deep-research-in-azure-ai-foundry-agent-service/)  
14. azure-ai-projects 1.0.0b6 \- PyPI, accessed on July 16, 2025, [https://pypi.org/project/azure-ai-projects/1.0.0b6/](https://pypi.org/project/azure-ai-projects/1.0.0b6/)  
15. azure-ai-projects 1.0.0b10 \- PyPI, accessed on July 16, 2025, [https://pypi.org/project/azure-ai-projects/1.0.0b10/](https://pypi.org/project/azure-ai-projects/1.0.0b10/)  
16. Azure AI Foundry Agent Service Deep Research Tool Error · Issue \#41935 \- GitHub, accessed on July 16, 2025, [https://github.com/azure/azure-sdk-for-python/issues/41935](https://github.com/azure/azure-sdk-for-python/issues/41935)  
17. Azure AI Agents client library for Python | Microsoft Learn, accessed on July 16, 2025, [https://learn.microsoft.com/en-us/python/api/overview/azure/ai-agents-readme?view=azure-python](https://learn.microsoft.com/en-us/python/api/overview/azure/ai-agents-readme?view=azure-python)  
18. Class DefaultAzureCredential | Azure SDK for .NET, accessed on July 16, 2025, [https://azuresdkdocs.z19.web.core.windows.net/dotnet/Azure.Identity/1.13.2/api/Azure.Identity/Azure.Identity.DefaultAzureCredential.html](https://azuresdkdocs.z19.web.core.windows.net/dotnet/Azure.Identity/1.13.2/api/Azure.Identity/Azure.Identity.DefaultAzureCredential.html)  
19. DefaultAzureCredential \- azure-identity 1.11.0 javadoc, accessed on July 16, 2025, [https://javadoc.io/doc/com.azure/azure-identity/1.11.0/com/azure/identity/DefaultAzureCredential.html](https://javadoc.io/doc/com.azure/azure-identity/1.11.0/com/azure/identity/DefaultAzureCredential.html)  
20. Authentication, accessed on July 16, 2025, [https://microsoft.github.io/spring-cloud-azure/4.0.0-beta.3/4.0.0-beta.3/reference/html/authentication.html](https://microsoft.github.io/spring-cloud-azure/4.0.0-beta.3/4.0.0-beta.3/reference/html/authentication.html)  
21. DefaultAzureCredential Class (Azure.Identity) \- Azure for .NET Developers | Microsoft Learn, accessed on July 16, 2025, [https://learn.microsoft.com/en-us/dotnet/api/azure.identity.defaultazurecredential?view=azure-dotnet](https://learn.microsoft.com/en-us/dotnet/api/azure.identity.defaultazurecredential?view=azure-dotnet)  
22. Lifespan Events \- FastAPI, accessed on July 16, 2025, [https://fastapi.tiangolo.com/advanced/events/](https://fastapi.tiangolo.com/advanced/events/)  
23. openwallet-foundation-labs/mcp-over-tsp-python: Python SDK for Model Context Protocol servers and clients with TSP transport \- GitHub, accessed on July 16, 2025, [https://github.com/openwallet-foundation-labs/mcp-over-tsp-python](https://github.com/openwallet-foundation-labs/mcp-over-tsp-python)  
24. MCP Context \- FastMCP, accessed on July 16, 2025, [https://gofastmcp.com/servers/context](https://gofastmcp.com/servers/context)  
25. Model Context Protocol \- Explained\! (with Python example) \- YouTube, accessed on July 16, 2025, [https://www.youtube.com/watch?v=JF14z6XO4Ho\&pp=0gcJCfwAo7VqN5tD](https://www.youtube.com/watch?v=JF14z6XO4Ho&pp=0gcJCfwAo7VqN5tD)  
26. MCP \- Model Context Protocol \- SDK \- Python \- YouTube, accessed on July 16, 2025, [https://www.youtube.com/watch?v=oq3dkNm51qc](https://www.youtube.com/watch?v=oq3dkNm51qc)  
27. azure.ai.projects.aio package — Azure SDK for Python 2.0.0 ... \- NET, accessed on July 16, 2025, [https://azuresdkdocs.z19.web.core.windows.net/python/azure-ai-projects/1.0.0b6/azure.ai.projects.aio.html](https://azuresdkdocs.z19.web.core.windows.net/python/azure-ai-projects/1.0.0b6/azure.ai.projects.aio.html)