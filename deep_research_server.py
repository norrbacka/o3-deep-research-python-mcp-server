#!/usr/bin/env python3
"""
Azure Deep Research MCP Server

This MCP server integrates with Azure AI Foundry's O3 Deep Research to perform
in-depth research queries. It creates research summaries saved as markdown files
and returns file:// URIs to MCP clients.
"""

import os
import sys
import asyncio
from typing import Optional
from datetime import datetime
from pathlib import Path
import uuid
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not required if env vars are set

# Check for required packages
try:
    from mcp.server.fastmcp import FastMCP, Context
except ImportError:
    print("ERROR: MCP package not installed. Please run: pip install mcp[cli]", file=sys.stderr)
    sys.exit(1)

try:
    from azure.ai.projects.aio import AIProjectClient
    from azure.identity.aio import DefaultAzureCredential
    from azure.ai.agents.aio import AgentsClient
    from azure.ai.agents.models import MessageRole, ThreadMessage, DeepResearchTool
except ImportError as e:
    print(f"ERROR: Azure packages not installed. {e}", file=sys.stderr)
    print("Please run: pip install --pre --upgrade azure-ai-projects azure-identity aiohttp", file=sys.stderr)
    sys.exit(1)


# Globals for shared state (initialized in lifespan)
project_client: Optional[AIProjectClient] = None
agents_client: Optional[AgentsClient] = None
deep_research_tool: Optional[DeepResearchTool] = None
agent_id: Optional[str] = None

# Output directory for research summaries
OUTPUT_DIR = Path("research_summaries")
OUTPUT_DIR.mkdir(exist_ok=True)


async def fetch_and_update_progress(
    thread_id: str,
    agents_client: AgentsClient,
    context: Context,
    last_message_id: Optional[str] = None,
) -> Optional[str]:
    """Fetch latest agent response and update progress via MCP context."""
    response = await agents_client.messages.get_last_message_by_role(
        thread_id=thread_id,
        role=MessageRole.AGENT,
    )
    
    if not response or response.id == last_message_id:
        return last_message_id  # No new content
    
    # Log progress to MCP client
    await context.info("Agent response received.")
    
    # Extract text content
    if response.text_messages:
        text_content = "\n".join(t.text.value for t in response.text_messages)
        await context.info(f"Response preview: {text_content[:200]}...")
    
    # Log citations
    for ann in response.url_citation_annotations:
        await context.info(f"URL Citation: [{ann.url_citation.title}]({ann.url_citation.url})")
    
    return response.id


def save_research_summary(
    message: ThreadMessage,
    query: str,
) -> str:
    """Save research summary to markdown file and return file:// URI."""
    if not message:
        raise ValueError("No message content provided.")
    
    # Generate unique filename: timestamp + sanitized query + UUID
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    # More restrictive sanitization for security
    sanitized_query = "".join(c for c in query if c.isalnum() or c == " ")[:30].strip()
    sanitized_query = sanitized_query.replace(" ", "_")
    unique_id = uuid.uuid4().hex[:8]
    filename = f"{timestamp}_{sanitized_query}_{unique_id}.md"
    
    # Ensure file stays within OUTPUT_DIR
    filepath = OUTPUT_DIR / filename
    filepath = filepath.resolve()
    if not str(filepath).startswith(str(OUTPUT_DIR.resolve())):
        raise ValueError("Invalid file path")
    
    # Create file with restricted permissions (owner read/write only)
    with open(filepath, "w", encoding="utf-8") as fp:
        # Write header
        fp.write(f"# Deep Research Summary\n\n")
        fp.write(f"**Query:** {query}\n")
        fp.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        fp.write("---\n\n")
        
        # Write text summary
        if message.text_messages:
            text_summary = "\n\n".join([t.text.value.strip() for t in message.text_messages])
            fp.write(text_summary)
        
        # Write unique URL citations
        if message.url_citation_annotations:
            fp.write("\n\n## References\n\n")
            seen_urls = set()
            for ann in message.url_citation_annotations:
                url = ann.url_citation.url
                title = ann.url_citation.title or url
                if url not in seen_urls:
                    fp.write(f"- [{title}]({url})\n")
                    seen_urls.add(url)
    
    # Return file:// URI for standard file access
    return filepath.resolve().as_uri()


@asynccontextmanager
async def server_lifespan(mcp: FastMCP) -> AsyncIterator[None]:
    """Manage server lifecycle: initialize Azure resources on startup, cleanup on shutdown."""
    global project_client, agents_client, deep_research_tool, agent_id
    
    # Initialize Azure clients
    project_client = AIProjectClient(
        endpoint=os.environ["PROJECT_ENDPOINT"],
        credential=DefaultAzureCredential(),
    )
    
    try:
        # Enter project context first
        await project_client.__aenter__()
        
        # Get Bing connection ID
        bing_connection = await project_client.connections.get(
            name=os.environ["BING_RESOURCE_NAME"]
        )
        conn_id = bing_connection.id
        
        # Get agents client
        agents_client = project_client.agents
        
        # Create DeepResearchTool with proper connection
        deep_research_tool = DeepResearchTool(
            bing_grounding_connection_id=conn_id,
            deep_research_model=os.environ["DEEP_RESEARCH_MODEL_DEPLOYMENT_NAME"],
        )
        
        # Get the tool definitions from DeepResearchTool
        tools_config = deep_research_tool.definitions
        agent = await agents_client.create_agent(
            model=os.environ["MODEL_DEPLOYMENT_NAME"],
            name="deep-research-agent",
            instructions="You are a research assistant. When asked to research a topic, use the deep_research tool to perform comprehensive research and provide detailed information.",
            tools=tools_config,
        )
        agent_id = agent.id
        # Log agent creation (without exposing ID in production)
        if os.environ.get("DEBUG"):
            print(f"Created reusable agent, ID: {agent_id}", file=sys.stderr)
        
        yield  # Server runs here
        
    finally:
        # Cleanup on shutdown
        if agent_id and agents_client:
            try:
                await agents_client.delete_agent(agent_id)
                print("Deleted agent", file=sys.stderr)
            except Exception as e:
                print(f"Error deleting agent: {e}", file=sys.stderr)
        
        if project_client:
            try:
                await project_client.__aexit__(None, None, None)
            except Exception as e:
                print(f"Error closing project client: {e}", file=sys.stderr)


# Create MCP server
mcp = FastMCP("AzureDeepResearchMCP", lifespan=server_lifespan)


@mcp.tool()
async def deep_research(context: Context, query: str) -> str:
    """
    Perform deep research on a topic using Azure AI Foundry and return path to summary Markdown file.
    
    Args:
        query: The research query to process
        
    Returns:
        File URI pointing to the generated research summary markdown file
    """
    # Input validation
    if not query or not isinstance(query, str):
        raise ValueError("Query must be a non-empty string")
    
    query = query.strip()
    if len(query) > 1000:  # Reasonable limit for research queries
        raise ValueError("Query too long. Maximum 1000 characters allowed")
    
    if len(query) < 10:
        raise ValueError("Query too short. Please provide a more detailed research question")
    if not agent_id or not agents_client:
        raise RuntimeError("Server not initialized properly. Check lifespan.")
    
    await context.info(f"Starting deep research for query: {query}")
    
    # Return immediately with a status file that will be updated
    status_file = OUTPUT_DIR / f"research_status_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    
    # Create initial status file with restricted permissions
    with open(status_file, "w") as f:
        f.write(f"# Deep Research Status\n\n")
        f.write(f"**Query:** {query}\n")
        f.write(f"**Started:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Status:** 🔄 Research initiated...\n\n")
        f.write(f"## Progress\n\n")
        f.write(f"Deep research has been started. This process can take up to 30 minutes.\n\n")
        f.write(f"Please check back later for the complete research summary.\n")
    
    # Start the research in the background
    asyncio.create_task(perform_deep_research_async(query, status_file, context))
    
    # Return immediately with the status file
    return status_file.resolve().as_uri()


async def perform_deep_research_async(query: str, status_file: Path, context: Context) -> None:
    """Perform the actual deep research in the background"""
    
    def update_status(status: str, progress: str = ""):
        """Update the status file with current progress"""
        with open(status_file, "w") as f:
            f.write(f"# Deep Research Status\n\n")
            f.write(f"**Query:** {query}\n")
            f.write(f"**Started:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Status:** {status}\n\n")
            f.write(f"## Progress\n\n")
            f.write(f"{progress}\n")
    
    try:
        # Update status
        update_status("🔄 Creating research thread...", "Setting up Azure AI agents...")
        
        # Create thread and message
        thread = await agents_client.threads.create()
        await context.info(f"Created thread: {thread.id}")
        
        message = await agents_client.messages.create(
            thread_id=thread.id,
            role="user",
            content=query,
        )
        await context.info(f"Created message: {message.id}")
        
        # Start run
        run = await agents_client.runs.create(
            thread_id=thread.id,
            agent_id=agent_id
        )
        await context.info("Research run started. Polling for completion...")
        update_status("🔄 Research started", "Initial processing...")
        
        # Poll for completion with progress updates
        last_message_id = None
        poll_count = 0
        max_polls = 1800  # 30 minutes at 1 second intervals
        
        while run.status in ("queued", "in_progress") and poll_count < max_polls:
            await asyncio.sleep(1)  # Non-blocking sleep
            poll_count += 1
            
            # Update run status
            run = await agents_client.runs.get(
                thread_id=thread.id,
                run_id=run.id
            )
            
            # Calculate elapsed time
            elapsed_minutes = poll_count // 60
            elapsed_seconds = poll_count % 60
            time_str = f"{elapsed_minutes}m {elapsed_seconds}s" if elapsed_minutes > 0 else f"{elapsed_seconds}s"
            
            # Update status file every 10 seconds
            if poll_count % 10 == 0:
                progress_text = f"Research in progress...\n"
                progress_text += f"- Elapsed time: {time_str}\n"
                progress_text += f"- Status: {run.status}\n"
                progress_text += f"- Progress: {poll_count}/{max_polls} seconds\n"
                update_status(f"🔄 Research {run.status}", progress_text)
                await context.info(f"Still polling... {poll_count} seconds elapsed")
            
            # Check for new messages
            last_message_id = await fetch_and_update_progress(
                thread_id=thread.id,
                agents_client=agents_client,
                context=context,
                last_message_id=last_message_id,
            )
        
        await context.info(f"Run finished with status: {run.status}")
        
        # Handle failure
        if run.status == "failed":
            error_msg = getattr(run, 'last_error', None) or "Unknown error"
            await context.error(f"Research failed: {error_msg}")
            raise ValueError(f"Deep research failed: {error_msg}")
        
        if poll_count >= max_polls:
            await context.error("Research timed out after 30 minutes")
            raise TimeoutError("Deep research timed out after 30 minutes")
        
        # Get the agent's response
        agent_response = await agents_client.messages.get_last_message_by_role(
            thread_id=thread.id,
            role=MessageRole.AGENT
        )
        
        if not agent_response:
            raise ValueError("No response from agent.")
        
        # Check if this is a clarification request (typically ends with a question mark)
        response_text = ""
        if agent_response.text_messages:
            response_text = " ".join(t.text.value for t in agent_response.text_messages)
        
        # If it's a clarification request, automatically respond with "go ahead"
        if response_text and ("?" in response_text or "Would you like" in response_text):
            await context.info("Agent is asking for clarification. Responding with 'go ahead'...")
            update_status("🔄 Handling clarification request", "Agent asked for clarification. Sending 'go ahead' to start deep research...")
            
            # Send "go ahead" message
            await agents_client.messages.create(
                thread_id=thread.id,
                role="user",
                content="go ahead"
            )
            
            # Create a new run to continue the research
            run = await agents_client.runs.create(
                thread_id=thread.id,
                agent_id=agent_id
            )
            await context.info("Continuing with deep research...")
            update_status("🔬 Deep research started", "The deep research process has begun. This typically takes 10-30 minutes to complete.")
            
            # Poll again for the actual research results
            poll_count = 0
            while run.status in ("queued", "in_progress") and poll_count < max_polls:
                await asyncio.sleep(1)
                poll_count += 1
                
                run = await agents_client.runs.get(
                    thread_id=thread.id,
                    run_id=run.id
                )
                
                # Calculate elapsed time
                elapsed_minutes = poll_count // 60
                elapsed_seconds = poll_count % 60
                time_str = f"{elapsed_minutes}m {elapsed_seconds}s" if elapsed_minutes > 0 else f"{elapsed_seconds}s"
                
                # Calculate percentage
                percentage = (poll_count / max_polls) * 100
                
                # Update status file every 30 seconds
                if poll_count % 30 == 0:
                    progress_text = f"Deep research analysis in progress...\n"
                    progress_text += f"- Elapsed time: {time_str}\n"
                    progress_text += f"- Progress: {percentage:.1f}%\n"
                    progress_text += f"- Estimated time remaining: {(max_polls - poll_count) // 60} minutes\n"
                    progress_text += f"\nThe AI is analyzing multiple sources and synthesizing information.\n"
                    update_status("🔬 Deep research analyzing", progress_text)
                
                last_message_id = await fetch_and_update_progress(
                    thread_id=thread.id,
                    agents_client=agents_client,
                    context=context,
                    last_message_id=last_message_id,
                )
                
                # Log every 30 seconds for deep research
                if poll_count % 30 == 0:
                    await context.info(f"Deep research in progress... {poll_count} seconds elapsed")
            
            # Get the final research response
            agent_response = await agents_client.messages.get_last_message_by_role(
                thread_id=thread.id,
                role=MessageRole.AGENT
            )
            
            if not agent_response:
                raise ValueError("No final research response from agent.")
        
        # Save final research summary
        if agent_response and agent_response.text_messages:
            # Replace the status file with the final research
            with open(status_file, "w") as f:
                f.write(f"# Deep Research Summary\n\n")
                f.write(f"**Query:** {query}\n")
                f.write(f"**Completed:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"**Status:** ✅ Research completed\n\n")
                f.write("---\n\n")
                
                # Write the research content
                text_summary = "\n\n".join([t.text.value.strip() for t in agent_response.text_messages])
                f.write(text_summary)
                
                # Add citations if available
                if agent_response.url_citation_annotations:
                    f.write("\n\n## References\n\n")
                    seen_urls = set()
                    for ann in agent_response.url_citation_annotations:
                        url = ann.url_citation.url
                        title = ann.url_citation.title or url
                        if url not in seen_urls:
                            f.write(f"- [{title}]({url})\n")
                            seen_urls.add(url)
            
            await context.info(f"Research completed and saved to: {status_file.resolve().as_uri()}")
        else:
            update_status("❌ Research failed", "No response received from the research agent.")
            
    except Exception as e:
        error_msg = f"Error during deep research: {str(e)}"
        await context.error(error_msg)
        update_status("❌ Research failed", error_msg)


@mcp.tool()
async def check_research_status(context: Context) -> str:
    """
    Check the status of all ongoing research tasks.
    
    Returns:
        Summary of research tasks and their current status
    """
    status_files = list(OUTPUT_DIR.glob("research_status_*.md"))
    
    if not status_files:
        return "No research tasks found."
    
    # Sort by modification time (most recent first)
    status_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    
    status_summary = "# Research Task Status\n\n"
    
    for status_file in status_files[:10]:  # Show last 10
        try:
            with open(status_file, "r") as f:
                content = f.read()
                # Extract key info
                lines = content.split("\n")
                query_line = next((l for l in lines if l.startswith("**Query:**")), "")
                status_line = next((l for l in lines if l.startswith("**Status:**")), "")
                
                status_summary += f"## {status_file.name}\n"
                status_summary += f"{query_line}\n"
                status_summary += f"{status_line}\n"
                status_summary += f"File: {status_file.resolve().as_uri()}\n\n"
        except Exception as e:
            status_summary += f"## {status_file.name}\n"
            status_summary += f"Error reading file: {e}\n\n"
    
    return status_summary


@mcp.prompt()
def generate_research_query(topic: str) -> str:
    """
    Generate a refined research query for deep research.
    
    Args:
        topic: The topic to research
        
    Returns:
        A well-structured research query
    """
    return f"Provide the latest in-depth research summary on {topic} from the past year, including key papers, findings, and emerging trends."


def main():
    """Main entry point for the server."""
    # Check for required environment variables
    required_vars = [
        "PROJECT_ENDPOINT",
        "BING_RESOURCE_NAME", 
        "DEEP_RESEARCH_MODEL_DEPLOYMENT_NAME",
        "MODEL_DEPLOYMENT_NAME"
    ]
    
    missing_vars = [var for var in required_vars if not os.environ.get(var)]
    if missing_vars:
        print("ERROR: Missing required environment variables:", file=sys.stderr)
        for var in missing_vars:
            print(f"  - {var}", file=sys.stderr)
        print("\nPlease set these in your .env file or environment.", file=sys.stderr)
        sys.exit(1)
    
    # Run the server
    print("Starting Azure Deep Research MCP Server...", file=sys.stderr)
    print(f"Output directory: {OUTPUT_DIR.absolute()}", file=sys.stderr)
    mcp.run(transport="stdio")  # For local testing with Claude Desktop
    # For production: mcp.run(transport="http", host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()