import asyncio
from collections.abc import AsyncGenerator
import os
import urllib.parse
from dotenv import load_dotenv
import streamlit as st
import urllib
from client import AgentClient, AgentClientError
from streamlit.runtime.scriptrunner import get_script_run_ctx

from schema import ChatHistory
from schema.schema import ChatMessage

APP_TITLE = "LangGraph Agents"
APP_ICON = "🧰"

async def main() -> None:
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon=APP_ICON,
        menu_items={}
    )
    
    if "agent_client" not in st.session_state:
        load_dotenv()
        agent_url = os.getenv("AGENT_URL")
        if not agent_url:
            host = os.getenv("HOST", "127.0.0.1")
            port = os.getenv("PORT", "8000")
            agent_url = f"http://{host}:{port}"
        try:
            with st.spinner("Connecting to agent service..."):
                st.session_state.agent_client = AgentClient(base_url=agent_url)
        except AgentClientError as e:
            st.error(f"Error connecting to agent service at {agent_url}: {e}")
            st.markdown("The service might be booting up. Try again in a few seconds.")
            st.stop()
    agent_client: AgentClient = st.session_state.agent_client
    
    if "thread_id" not in st.session_state:
        thread_id = st.query_params.get("thread_id")
        if not thread_id:
            thread_id = get_script_run_ctx().session_id
            messages = []
        else:
            try:
                messages: ChatHistory = agent_client.get_history(thread_id=thread_id).messages
            except AgentClientError:
                st.error("No message history found for this Thread ID.")
                messages = []
        st.session_state.messages = messages
        st.session_state.thread_id = thread_id
        
    # Config options
    with st.sidebar:
        st.header(f"{APP_ICON} {APP_TITLE}")
        ""
        "Agentic AI for performing various tasks using LangGraph, FastAPI and Streamlit"
        with st.popover(":material/settings: Settings", use_container_width=True):
            model_idx = agent_client.info.models.index(agent_client.info.default_model)
            model = st.selectbox("LLM to use", options=agent_client.info.models, index=model_idx)
            agent_list = [a.key for a in agent_client.info.agents]
            agent_idx = agent_list.index(agent_client.info.default_agent)
            agent_client.agent = st.selectbox(
                "Agent to use",
                options=agent_list,
                index=agent_idx
            )
            use_streaming = st.toggle("Stream results", value=True)
        
        @st.dialog("Architecture")
        def architecture_dialog() -> None:
            st.image(
                os.path.join(os.getcwd(),"media/graph.png"),
            )
            "[View full size on Github]()"
            st.caption(
                "App hosted by Soumadip locally."
            )
        
        if st.button(":material/schema: Architecture", use_container_width=True):
            architecture_dialog()
        
        with st.popover(":material/policy: Privacy", use_container_width=True):
            st.write(
                "EVERYTHING WILL BE STORED AND MONITORED FOR PERSONAL GAIN SO, **FUCK YOU!!!!!**"
            )
        
        @st.dialog("Share/resume chat")
        def share_chat_dialog() -> None:
            session = st.runtime.get_instance()._session_mgr.list_active_sessions()[0]
            st_base_url = urllib.parse.urlunparse(
                [session.client.request.protocol, session.client.request.host, "", "", "", ""]
            )
            # If it is not localhost switch to https by default
            if not st_base_url.startswith("https") and "localhost" not in st_base_url:
                st_base_url = st_base_url.replace("http", "https")
            chat_url = f"{st_base_url}/?thread_id={st.session_state.thread_id}"
            st.markdown(f"**Chat URL:**\n```text\n{chat_url}\n```")
            st.info("Copy the URL to share or revisit this chat")
        
        if st.button(":material/upload: Share/resume chat", use_container_width=True):
            share_chat_dialog()
        
    # Draw existing messages
    messages: list[ChatMessage] = st.session_state.messages
    
    if len(messages) == 0:
        WELCOME = "Hello! I'm an AI powered research assistant with web search, created by *SOUMADIP SAHA*. Ask me anything!"
        with st.chat_message("ai"):
            st.write(WELCOME)
        
    # draw_messages() expects an async iterator over messages
    async def amessage_iter() -> AsyncGenerator[ChatMessage, None]:
        for m in messages:
            yield m
    
    await draw_messages(amessage_iter())

    # Generate new message if the user provided new input
    if user_input := st.chat_input():
        messages.append(ChatMessage(type="human", content=user_input))
        st.chat_message("human").write(user_input)
        try:
            if use_streaming:
                stream = agent_client.astream(
                    message=user_input,
                    model=model,
                    thread_id=st.session_state.thread_id
                )
                await draw_messages(stream, is_new=True)
            else:
                response = await agent_client.ainvoke(
                    message=user_input,
                    model=model,
                    thread_id=st.session_state.thread_id
                )
                messages.append(response)
                st.chat_message("ai").write(response.content)
            st.rerun()  # Clear state containers
        except AgentClientError as e:
            st.error(f"Error generating response: {e}")
            st.stop()

async def draw_messages(
    messages_agen: AsyncGenerator[ChatMessage | str, None],
    is_new: bool = False
) -> None:
    """
    Draw a set of chat messages - either replaying existing messages
    or streaming new ones.
    
    This function has additional logic to handle streaming tokens and tool calls.
    - Use a placeholder container to render streaming tokens as they arrive.
    - Use a status container to render tool calls. Track the tool inputs and outputs
      and update the status container accordingly.
      
    The function also needs to track the last message container in session state
    since later messages can draw to the same container.
    
    Args:
        messages_agen: An async iterator over messages to draw.
        is_new: Whether the messages are new or not.
    """
    
    # keep track of the last message container
    last_message_type = None
    st.session_state.last_message = None
    
    # Placeholder for intermediate streaming tokens
    streaming_content = ""
    streaming_placeholder = None
    
    # Iterate over the messages and draw them
    while msg := await anext(messages_agen, None):
        # str message represents an intermediate token being streamed
        if isinstance(msg, str):
            # If placeholder is empty, this is the first token of a new message
            # being streamed. We need to do setup.
            if not streaming_placeholder:
                if last_message_type != "ai":
                    last_message_type = "ai"
                    st.session_state.last_message = st.chat_message("ai")
                with st.session_state.last_message:
                    streaming_placeholder = st.empty()

            streaming_content += msg
            streaming_placeholder.markdown(streaming_content)   # Change if needed to st.write
            continue
        if not isinstance(msg, ChatMessage):
            st.error(f"Unexpected message type: {type(msg)}")
            st.write(msg)
            st.stop()
        match msg.type:
            # A message from the user, the easiest case
            case "human":
                last_message_type = "human"
                st.chat_message("human").write(msg.content)
            
            # A message from the agent is the most complex case, since we need to
            # handle streaming tokens and tool calls.
            case "ai":
                # If we're rendering new messages, store the message in session state
                if is_new:
                    st.session_state.messages.append(msg)
                
                # If the last message was not AI, creatge a new chat message
                if last_message_type != "ai":
                    last_message_type = "ai"
                    st.session_state.last_message = st.chat_message("ai")
                
                with st.session_state.last_message:
                    # If the message has content, write it out.
                    # Reset the streaming variables to prepare for the next message.
                    if msg.content:
                        if streaming_placeholder:
                            streaming_placeholder.markdown(msg.content)      # Change if needed to st.write
                            streaming_content = ""
                            streaming_placeholder = None
                        else:
                            st.markdown(msg.content)        # Change if needed to st.write
                    
                    if msg.tool_calls:
                        # Create a status container for each tool call and store the 
                        # status container by ID to ensure results are mapped to the
                        # correct status container.
                        call_results = {}
                        for tool_call in msg.tool_calls:
                            status = st.status(
                                f"""Tool Call: {tool_call["name"]}""",
                                state="running" if is_new else "complete",
                            )
                            call_results[tool_call["id"]] = status
                            status.markdown("Input:")        # Change if needed to st.write
                            status.markdown(tool_call["args"])      # Change if needed to st.write
                        
                        # Expect one ToolMessage for each tool call.
                        for _ in range(len(call_results)):
                            tool_result: ChatMessage = await anext(messages_agen)
                            if tool_result.type != "tool":
                                st.error(f"Unexpected ChatMessage type: {tool_result.type}")
                                st.write(tool_result)
                                st.stop()
                                
                        # Record the message if it's new, update the correct
                        # status container with the result
                        if is_new:
                            st.session_state.messages.append(tool_result)
                        status = call_results[tool_result.tool_call_id]
                        status.markdown("Output:")      # Change if needed to st.write
                        status.markdown(tool_result.content)     # Change if needed to st.write
                        status.update(state="complete")
            
            case "custom":
                # TODO: Handle custom messages
                st.error(f"Custom message type: {msg.type}")
                st.markdown(msg.content)        # Change if needed to st.write
                st.stop()
            
            case _:
                st.error(f"Unexpected message type: {msg.type}")
                st.markdown(msg)
                st.stop()

if __name__ == "__main__":
    asyncio.run(main())