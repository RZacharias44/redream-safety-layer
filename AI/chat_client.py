import asyncio
import aiohttp
import sys
import uuid
import json
import logging
import argparse
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Function to load history and send it to the API
async def load_and_set_history(session, session_id, user_id, history_file):
    print(f"Loading history from {history_file} to set on server...")
    print("-" * 75)
    messages_to_send = []
    history_to_display = []
    try:
        with open(history_file, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            try:
                role, content = line.split(":", 1)
                role = role.strip().lower()
                content = content.strip()
                
                # Basic validation for role
                if role not in ["user", "assistant", "ai"]:
                     print(f"Skipping invalid role in history file: {role} (line: {line})")
                     continue
                
                # Standardize role to 'user' or 'assistant'
                api_role = "user" if role == "user" else "assistant"
                display_role = role.capitalize()

                # Create message dictionary matching the server's Message model (approximated)
                # We don't have stage/language info here, server will handle defaults
                message_dict = {
                    "role": api_role,
                    "content": content,
                    # Optional: Add timestamp if needed/available, defaults on server
                    "timestamp": datetime.now().isoformat() 
                }
                messages_to_send.append(message_dict)
                history_to_display.append((display_role, content))

            except ValueError:
                print(f"Skipping invalid line format in history file: {line}")
                continue
        
        # Send history to the server endpoint
        if messages_to_send:
            request_data = {
                "session_id": session_id,
                "user_id": user_id,
                "messages": messages_to_send
                # We could add optional stages/language here if parsed from file
            }
            try:
                async with session.post(
                    "http://localhost:8000/chat/set_history", 
                    json=request_data,
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as response:
                    if response.status == 200:
                        response_data = await response.json()
                        print(f"Successfully set history on server for session {response_data.get('session_id', session_id)[:8]}. Messages loaded: {len(messages_to_send)}")
                    else:
                        error_text = await response.text()
                        print(f"\nError setting history on server: Status {response.status} - {error_text}")
                        print("Continuing without setting history...")
                        history_to_display = [] # Clear display if setting failed
                        messages_to_send = [] # Clear messages if setting failed

            except aiohttp.ClientError as e:
                print(f"\nConnection error setting history: {str(e)}")
                print("Continuing without setting history...")
                history_to_display = [] # Clear display if setting failed
                messages_to_send = [] # Clear messages if setting failed
        else:
             print("No valid messages found in history file to send.")
             history_to_display = []

        # Display the history that was attempted to be loaded
        if history_to_display:
            print("-" * 75)
            print("Loaded History:")
            print("-" * 75)
            for role, msg in history_to_display:
                 print(f"{role}: {msg}")
                 print("-" * 75)
            print("Continuing chat...")
            print("-" * 75)
        
        # Return the last user message content if history load was successful and it ended with user input
        if messages_to_send and messages_to_send[-1]["role"] == "user":
            return messages_to_send[-1]["content"]
        else:
            return None

    except FileNotFoundError:
        print(f"Error: History file not found: {history_file}")
        return None # Ensure None is returned on error
    except Exception as e:
        print(f"Error reading or processing history file: {str(e)}")
        return None # Ensure None is returned on error


async def stream_chat(session_id=None, user_id=None, history_file=None):
    """Streaming chat client"""
    session_id = session_id or str(uuid.uuid4())
    user_id = user_id or str(uuid.uuid4())
    print("\nStreaming Chat initialized. Type 'quit' to exit.")
    print(f"Session ID: {session_id}")
    print(f"User ID: {user_id}")
    print("-" * 75)

    intro_message_de = "AI: Ich bin hier, um dir zu helfen, deine Albträume zu bewältigen und sie in positivere Erfahrungen zu verwandeln.\nNimm dir Zeit, deinen Albtraum so detailliert wie möglich zu beschreiben. Wenn du fertig bist, werde ich hier sein, um dir Anleitung und Unterstützung zu bieten, während wir ihn gemeinsam in eine positivere Erzählung umwandeln."
    print(intro_message_de)
    print("-" * 75)
    
    timeout = aiohttp.ClientTimeout(total=None)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        # Load and set history if provided
        last_user_msg_from_history = None
        if history_file:
            last_user_msg_from_history = await load_and_set_history(session, session_id, user_id, history_file)

        # If history ended with a user message, get the AI's response first
        if last_user_msg_from_history:
            print("AI: ", end="", flush=True) # Prompt for AI response
            try:
                for attempt in range(3):
                    try:
                        async with session.post(
                            "http://localhost:8000/chat/stream",
                            json={
                                "session_id": session_id,
                                "message": last_user_msg_from_history, # Send last user message
                                "user_id": user_id
                            }
                        ) as response:
                            if response.status != 200:
                                print(f"\nError getting initial AI response: Server returned status {response.status}")
                                continue
                            
                            initial_stages = []
                            initial_language = None
                            async for line in response.content:
                                line = line.decode('utf-8')
                                if line.startswith("data: "):
                                    data = line[6:]
                                    if data.strip() == "[DONE]":
                                        break
                                    try:
                                        chunk = json.loads(data)
                                        if "content" in chunk:
                                            print(chunk["content"], end="", flush=True)
                                        if "stages" in chunk:
                                            initial_stages = chunk["stages"]
                                        if "language" in chunk:
                                            initial_language = chunk["language"]
                                    except json.JSONDecodeError:
                                        print(f"\nError decoding initial AI response: {data}")
                            break # Successful completion

                        # Print metadata for the initial response
                        print(f"\n| Stages: {' → '.join(initial_stages)}")
                        if initial_language:
                            print(f"| Language: {initial_language}")
                        print("-" * 75)

                    except (aiohttp.ClientError, json.JSONDecodeError) as e:
                        if attempt < 2: 
                            print(f"\nConnection error getting initial AI response (retry {attempt+1}/3): {str(e)}")
                            await asyncio.sleep(1)
                        else:
                            print(f"\nFailed to get initial AI response after multiple retries: {str(e)}")
                            # Decide if we should proceed or exit? For now, proceed.
                            break # Break retry loop on final failure
                
                # Ensure newline and separator after initial response before asking for input
                print()
                print("-" * 75)
            except Exception as e:
                print(f"\nError getting initial AI response: {str(e)}")
                print("-" * 75)
                print()
                print("-" * 75)

        # Now start the regular interactive loop
        while True:
            message = input("User: ").strip()
            print("-" * 75)
            
            if message.lower() == 'quit':
                break
                
            try:
                for attempt in range(3): 
                    try:
                        async with session.post(
                            "http://localhost:8000/chat/stream",
                            json={
                                "session_id": session_id,
                                "message": message,
                                "user_id": user_id
                            }
                        ) as response:
                            if response.status != 200:
                                print(f"Error: Server returned status {response.status}")
                                continue

                            print("AI: ", end="", flush=True)
                            stages = []
                            language = None
                            async for line in response.content:
                                line = line.decode('utf-8')
                                if line.startswith("data: "):
                                    data = line[6:]
                                    if data.strip() == "[DONE]":
                                        break
                                    try:
                                        chunk = json.loads(data)
                                        if "content" in chunk:
                                            print(chunk["content"], end="", flush=True)
                                        if "stages" in chunk:
                                            stages = chunk["stages"]
                                        if "language" in chunk:
                                            language = chunk["language"]
                                    except json.JSONDecodeError:
                                        print(f"\nError decoding response: {data}")
                            break  # Successful completion

                    except (aiohttp.ClientError, json.JSONDecodeError) as e:
                        if attempt < 2: 
                            print(f"Connection error (retry {attempt+1}/3): {str(e)}")
                            await asyncio.sleep(1)
                        else:
                            raise

                print(f"\n| Stages: {' → '.join(stages)}")
                if language:
                    print(f"| Language: {language}")
                print("-" * 75)
                # Ensure newline and separator after streaming response before asking for input again
                print()
                print("-" * 75)

            except Exception as e:
                print(f"Error: {str(e)}")
                print("-" * 75)
                print()
                print("-" * 75)

async def regular_chat(session_id=None, user_id=None, history_file=None):
    """Non-streaming chat client"""
    session_id = session_id or str(uuid.uuid4())
    user_id = user_id or str(uuid.uuid4())
    print("\nChat initialized. Type 'quit' to exit.")
    print(f"Session ID: {session_id}")
    print(f"User ID: {user_id}")
    print("-" * 75)

    intro_message_de = "AI: Ich bin hier, um dir zu helfen, deine Albträume zu bewältigen und sie in positivere Erfahrungen zu verwandeln.\nNimm dir Zeit, deinen Albtraum so detailliert wie möglich zu beschreiben. Wenn du fertig bist, werde ich hier sein, um dir Anleitung und Unterstützung zu bieten, während wir ihn gemeinsam in eine positivere Erzählung umwandeln."
    print(intro_message_de)
    print("-" * 75)
    
    timeout = aiohttp.ClientTimeout(total=60)
    
    async def make_chat_request(session, message_text, retries=3): 
        for attempt in range(retries):
            try:
                request_data = {
                    "session_id": session_id,
                    "user_id": user_id,
                    "message": message_text
                }
                
                async with session.post(
                    "http://localhost:8000/chat",
                    json=request_data,
                    timeout=timeout
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    print(f"Server error (status {response.status}): {await response.text()}")
                    
            except aiohttp.ClientError as e:
                print(f"Connection error (attempt {attempt + 1}/{retries}): {str(e)}")
                if attempt < retries - 1:
                    await asyncio.sleep(1)
                else:
                    raise

    async with aiohttp.ClientSession() as session:
        # Load and set history if provided
        last_user_msg_from_history = None
        if history_file:
            last_user_msg_from_history = await load_and_set_history(session, session_id, user_id, history_file)

        # If history ended with a user message, get the AI's response first
        if last_user_msg_from_history:
            try:
                print("AI: Fetching response to last history message...")
                data = await make_chat_request(session, last_user_msg_from_history)
                if data:
                    print(f"AI: {data['response']}")
                    print(f"\n| Stages: {' → '.join(data['stages'])}")
                    if 'language' in data:
                        print(f"| Language: {data['language']}")
                else:
                    print("AI: No initial response received.")
                print("-" * 75)
            except Exception as e:
                print(f"\nError getting initial AI response: {str(e)}")
                print(f"Error type: {type(e)}")
                print("-" * 75)

        # Now start the regular interactive loop
        while True:
            message = input("User: ").strip()
            print("-" * 75)
            
            if message.lower() == 'quit':
                break
                
            try:
                data = await make_chat_request(session, message) 
                if data:
                    print(f"AI: {data['response']}")
                    print(f"\n| Stages: {' → '.join(data['stages'])}")
                    if 'language' in data:
                        print(f"| Language: {data['language']}")
                print("-" * 75)
                    
            except Exception as e:
                print(f"Error: {str(e)}")
                print(f"Error type: {type(e)}")
                print("-" * 75)

if __name__ == "__main__":
    # Setup argument parser
    parser = argparse.ArgumentParser(description="Chat client for DreamMend IRT Bot")
    parser.add_argument("--stream", action="store_true", help="Enable streaming mode.")
    parser.add_argument("--session-id", type=str, help="Use a specific session ID.")
    parser.add_argument("--user-id", type=str, help="Use a specific user ID.")
    parser.add_argument("--history-file", type=str, help="Path to a file containing conversation history to load (format: 'Role: Message' per line, e.g., 'User: Hello' or 'AI: Hi').")
    
    args = parser.parse_args()

    # Prepare arguments for the chat functions
    chat_args = {
        "session_id": args.session_id,
        "user_id": args.user_id,
        "history_file": args.history_file
    }

    # Start the appropriate chat mode
    if args.stream:
        asyncio.run(stream_chat(**chat_args))
    else:
        asyncio.run(regular_chat(**chat_args))

