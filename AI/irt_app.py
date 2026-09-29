from typing import Tuple, AsyncGenerator, Optional
import os
from dotenv import load_dotenv, find_dotenv
from .models import Conversation, Stage, ChatInput, ChatResponse
import logging
import json
from langfuse.decorators import observe, langfuse_context
from .prompts import (SYSTEM_PROMPT_TEMPLATES_DE, SYSTEM_PROMPT_TEMPLATES_EN,
                      NIGHTMARE_SUMMARY_PROMPT_DE, NIGHTMARE_SUMMARY_PROMPT_EN,
                      SCOPE_BOUNDARY_DE, SCOPE_BOUNDARY_EN)

# Load environment variables (searches upward from CWD to find project .env)
load_dotenv(find_dotenv())

# Language detection removed; language is controlled via override from client

# Import agents AFTER env is loaded so API keys are available during initialization
from .agent import routing_agent, response_agent  # noqa: E402

# Routing and chatbot responses use Scaleway-hosted Mistral with direct Mistral fallback.
if not os.getenv('SCALEWAY_API_KEY') or not os.getenv('MISTRAL_API_KEY'):
    raise ValueError("SCALEWAY_API_KEY and MISTRAL_API_KEY environment variables must be set")

# Get logger instances
logger = logging.getLogger(__name__)

# Initialize safety critic (fail-open: chatbot works without it)
try:
    from safety.critic import SafetyCritic, SafetyResult
    safety_critic = SafetyCritic()
    logger.info("SafetyCritic initialized successfully")
except Exception as e:
    safety_critic = None
    logger.warning(f"SafetyCritic initialization failed: {e}. Safety checks disabled.")

# Configure Langfuse after load_dotenv()
# Set LANGFUSE_ENABLED=false in .env or environment to disable tracing
langfuse_context.configure(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST"),
    enabled=os.getenv("LANGFUSE_ENABLED", "true").lower() != "false"
)

language = "de"

@observe(as_type="trace", capture_input=False, capture_output=False)
async def process_chat_message(
    chat_input: ChatInput,
    conversation: Conversation,
    safety_enabled: bool = True,
) -> ChatResponse:
    """Process a chat message and return complete response"""
    try:
        # Update the trace level input/output
        langfuse_context.update_current_trace(
            name=f"Chat Session: {chat_input.session_id[:8]}",
            session_id=chat_input.session_id,
            user_id=chat_input.user_id,
            tags=conversation.stages,
            input=chat_input.message,
            output=None
        )
        
        # Update observation level input/output
        langfuse_context.update_current_observation(
            name="Process Message",
            input=chat_input.message,
            output=None,
            metadata={
                "type": "chat_processing",
                "is_streaming": False,
                "user_id": chat_input.user_id
            }
        )
        
        # Add user message first so stage determination sees it in history
        # Use current conversation language or override for initial tagging
        initial_language = chat_input.language_override or conversation.language
        conversation.add_message(chat_input.message, "user", language=initial_language)

        # Language is controlled by override (from UI) or previous conversation setting
        detected_language = chat_input.language_override or conversation.language or "en"
        stage = await determine_stage_async(chat_input.message, conversation)
        conversation.language = detected_language

        # Generate nightmare summary at recording→rewriting transition
        if stage == "rewriting" and conversation.nightmare_summary is None:
            conversation.nightmare_summary = await generate_nightmare_summary(conversation)

        # Safety evaluation during rewriting stage
        safety_result = None
        if safety_enabled and stage == "rewriting":
            safety_result = await evaluate_safety(chat_input.message, conversation)

        response, usage = await get_response_async(
            stage, chat_input.message, conversation,
            safety_constraint=safety_result.constraint if safety_result else None
        )
        conversation.add_message(response, "assistant", stage, language=detected_language)

        response_obj = ChatResponse(
            session_id=chat_input.session_id,
            stage=stage,
            response=response,
            stages=conversation.stages,
            usage=usage,
            language=detected_language,
            safety=safety_result.to_dict() if safety_result else None
        )
        
        # Update both trace and observation output
        langfuse_context.update_current_trace(
            output=response
        )
        langfuse_context.update_current_observation(
            output=response,
            usage=usage
        )
        
        return response_obj
    except Exception as e:
        logger.error(f"Error processing message: {str(e)}")
        raise

@observe(as_type="trace", capture_input=False, capture_output=False)
async def process_chat_message_stream(
    chat_input: ChatInput,
    conversation: Conversation,
    safety_enabled: bool = True,
) -> AsyncGenerator[str, None]:
    """Process a chat message and yield streaming response"""
    try:
        # Update trace level with session, user, input
        langfuse_context.update_current_trace(
            name=f"Streaming Chat Session: {chat_input.session_id[:8]}",
            session_id=chat_input.session_id,
            user_id=chat_input.user_id,
            tags=conversation.stages,
            input=chat_input.message
        )
        
        langfuse_context.update_current_observation(
            name="Process Stream Message",
            input=chat_input.message,
            output=None,
            metadata={
                "type": "chat_processing",
                "is_streaming": True
            }
        )
        
        # Add user message first so stage determination sees it in history
        initial_language = chat_input.language_override or conversation.language
        conversation.add_message(chat_input.message, "user", language=initial_language)

        # Language is controlled by override (from UI) or previous conversation setting
        detected_language = chat_input.language_override or conversation.language or "en"
        stage = await determine_stage_async(chat_input.message, conversation)
        conversation.language = detected_language

        # Generate nightmare summary at recording→rewriting transition
        if stage == "rewriting" and conversation.nightmare_summary is None:
            conversation.nightmare_summary = await generate_nightmare_summary(conversation)

        # Safety evaluation during rewriting stage
        safety_result = None
        if safety_enabled and stage == "rewriting":
            safety_result = await evaluate_safety(chat_input.message, conversation)

        history = conversation.get_history_as_string()
        intro_message_de = "AI: Ich bin hier, um dir zu helfen, deine Albträume zu bewältigen und sie in positivere Erfahrungen zu verwandeln.\\nNimm dir Zeit, deinen Albtraum so detailliert wie möglich zu beschreiben. Wenn du fertig bist, werde ich hier sein, um dir Anleitung und Unterstützung zu bieten, während wir ihn gemeinsam in eine positivere Erzählung umwandeln."
        intro_message_en = "AI: I'm here to help you work through your nightmares and turn them into more positive experiences.\\nTake your time to describe your nightmare in as much detail as you can. When you're ready, I'll be here to guide and support you as we reshape it together into a more empowering story."

        is_english = (conversation.language == "en")
        templates = SYSTEM_PROMPT_TEMPLATES_EN if is_english else SYSTEM_PROMPT_TEMPLATES_DE
        intro_message = intro_message_en if is_english else intro_message_de

        # Prepend intro message to the history string used for the prompt
        history_for_prompt = intro_message + "\\n" + history if history else intro_message
        
        prompt_template = templates[stage]
        full_prompt = f"\n\nConversation history:\n{history_for_prompt}"
        
        full_response = ""
        final_usage = {}
        async for chunk, chunk_usage in get_response_stream_async(
            stage, full_prompt, conversation, detected_language,
            safety_constraint=safety_result.constraint if safety_result else None
        ):
            if chunk:
                full_response += chunk
                yield f"data: {json.dumps({'content': chunk, 'stages': conversation.stages, 'language': detected_language})}\n\n"
            if chunk_usage:
                final_usage = chunk_usage
        
        conversation.add_message(full_response, "assistant", stage, language=detected_language)
        
        # Update trace output at the end
        langfuse_context.update_current_trace(
            output=full_response
        )
        langfuse_context.update_current_observation(
            output=full_response,
            usage=final_usage
        )
        
        if safety_result:
            yield f"data: {json.dumps({'safety': safety_result.to_dict()})}\n\n"
        yield "data: [DONE]\n\n"

    except Exception as e:
        logger.error(f"Streaming error: {str(e)}")
        yield f"data: {json.dumps({'error': str(e)})}\n\n"

@observe(name="determine_stage", as_type="generation", capture_input=False, capture_output=False)
async def determine_stage_async(user_input: str, conversation: Conversation) -> str:
    """Async version of determine_stage"""
    langfuse_context.update_current_observation(
        name="Stage Determination",
        input=user_input,
        output=None,  # Will be set later
        metadata={"type": "stage_determination"}
    )
    
    history = conversation.get_history_as_string()
    # Corrected prompt: ROUTING_PROMPT is handled by the agent's system_prompt
    prompt = f"<transcript>\n{history}\n</transcript>\n\nClassification:"
    
    stage_response, usage, _ = await routing_agent.generate(prompt)  # Unpack content, usage, and logprobs
    stage_str = stage_response.strip()
    logger.info(f"Stage output: {stage_str}")
    
    try:
        stage = Stage(stage_str)
    except ValueError:
        print(f"Invalid stage {stage_str}, defaulting to last stage")
        stage = Stage(conversation.stages[-1])

    # *** START ÄNDERUNG ***
    # Diese 'Guard Rail' stellt sicher, dass die Stufen nicht übersprungen werden.
    if stage == Stage.FINAL:
        previous_stages = set(conversation.stages)
        
        # Guard rail 1: Muss eine Zusammenfassung haben, bevor irgendetwas anderes passiert
        # (Wenn keine Zusammenfassung vorhanden ist, leite zu 'summary' um)
        if Stage.SUMMARY.value not in previous_stages:
            print("Redirecting to summary stage as no summary has been generated yet")
            stage = Stage.SUMMARY
            
        # Guard rail 2: Muss ein Rehearsal gehabt haben, bevor es zum Abschluss kommt
        # (Wenn eine Zusammenfassung vorhanden ist, aber kein Rehearsal, leite zu 'rehearsal' um)
        elif Stage.REHEARSAL.value not in previous_stages:
            print("Redirecting to rehearsal stage as no rehearsal has been generated yet")
            stage = Stage.REHEARSAL
    # *** ENDE ÄNDERUNG ***
    
    conversation.stages.append(stage.value)
    
    langfuse_context.update_current_observation(
        output=stage.value,
        metadata={"stage": stage.value},
        usage=usage  # Add usage data to the observation
    )
    
    return stage.value

@observe(name="generate_nightmare_summary", as_type="generation")
async def generate_nightmare_summary(conversation: Conversation) -> Optional[str]:
    """Generate a comprehensive summary of the nightmare from the recording phase.

    Called once at the recording→rewriting transition. Extracts user messages
    from before the first rewriting-stage response and summarizes them.
    """
    try:
        is_english = (conversation.language == "en")
        summary_prompt = NIGHTMARE_SUMMARY_PROMPT_EN if is_english else NIGHTMARE_SUMMARY_PROMPT_DE

        # Collect user messages from recording phase (before first rewriting assistant message)
        recording_messages = []
        for msg in conversation.messages:
            if msg.role == "assistant" and msg.stage == "rewriting":
                break
            if msg.role == "user":
                recording_messages.append(msg.content)

        if not recording_messages:
            logger.warning("No recording-phase user messages found for nightmare summary")
            return None

        recording_text = "\n".join(recording_messages)
        prompt = f"Patient's nightmare description:\n{recording_text}"

        response_agent.system_prompt = summary_prompt
        summary, usage, _ = await response_agent.generate(prompt)

        logger.info(f"Generated nightmare summary ({len(summary)} chars)")
        return summary.strip()

    except Exception as e:
        logger.error(f"Failed to generate nightmare summary: {e}")
        return None


@observe(name="safety_evaluation", as_type="generation")
async def evaluate_safety(user_input: str, conversation: Conversation):
    """Run SafetyCritic on user input during the rewriting stage.

    Returns SafetyResult or None if the critic is unavailable or fails.
    Fail-open: the chatbot still has system prompt safety instructions.
    """
    if safety_critic is None:
        return None

    try:
        result = await safety_critic.evaluate_intervention(
            user_input,
            nightmare_context=conversation.nightmare_summary
        )
        logger.info(f"Safety evaluation: is_safe={result.is_safe}, node={result.node_id}, severity={result.severity}")
        return result
    except Exception as e:
        logger.error(f"Safety evaluation failed (proceeding without constraint): {e}")
        return None


@observe(name="get_response", as_type="generation", capture_input=False, capture_output=False)
async def get_response_async(stage: str, user_input: str, conversation: Conversation, safety_constraint: Optional[str] = None) -> Tuple[str, dict]:
    """Async version of get_response"""
    langfuse_context.update_current_observation(
        name="Response Generation",
        input=user_input,
        output=None,  # Will be set later
        metadata={
            "type": "response_generation",
            "stage": stage
        }
    )
    
    history = conversation.get_history_as_string()
    intro_message_de = "AI: Ich bin hier, um dir zu helfen, deine Albträume zu bewältigen und sie in positivere Erfahrungen zu verwandeln.\\nNimm dir Zeit, deinen Albtraum so detailliert wie möglich zu beschreiben. Wenn du fertig bist, werde ich hier sein, um dir Anleitung und Unterstützung zu bieten, während wir ihn gemeinsam in eine positivere Erzählung umwandeln."
    intro_message_en = "AI: I'm here to help you work through your nightmares and turn them into more positive experiences.\\nTake your time to describe your nightmare in as much detail as you can. When you're ready, I'll be here to guide and support you as we reshape it together into a more empowering story."

    is_english = (conversation.language == "en")
    templates = SYSTEM_PROMPT_TEMPLATES_EN if is_english else SYSTEM_PROMPT_TEMPLATES_DE
    intro_message = intro_message_en if is_english else intro_message_de

    # Prepend intro message to the history string used for the prompt
    history_for_prompt = intro_message + "\\n" + history if history else intro_message
    
    prompt_template = templates[stage]
    full_prompt = f"\n\nConversation history:\n{history_for_prompt}" # Use the modified history

    boundary = SCOPE_BOUNDARY_EN if is_english else SCOPE_BOUNDARY_DE
    base_prompt = boundary + "\n\n" + prompt_template
    if safety_constraint:
        response_agent.system_prompt = base_prompt + "\n\n" + safety_constraint
    else:
        response_agent.system_prompt = base_prompt
    response, usage, _ = await response_agent.generate(full_prompt)
    
    langfuse_context.update_current_observation(
        output=response,
        usage=usage
    )
    
    return response, usage

@observe(name="get_response_stream", as_type="generation", capture_input=False, capture_output=False)
async def get_response_stream_async(stage: str, full_prompt: str, conversation: Conversation, language: str, safety_constraint: Optional[str] = None) -> AsyncGenerator[Tuple[str, Optional[dict]], None]:
    """Async streaming version of get_response"""
    langfuse_context.update_current_observation(
        name="Response Generation",
        input=full_prompt,
        output=None,  # Will be set after streaming completes
        metadata={
            "type": "response_generation",
            "stage": stage,
            "language": language,
            "is_streaming": True
        }
    )
    
    is_english = (language == "en")
    templates = SYSTEM_PROMPT_TEMPLATES_EN if is_english else SYSTEM_PROMPT_TEMPLATES_DE
    prompt_template = templates[stage]
    boundary = SCOPE_BOUNDARY_EN if is_english else SCOPE_BOUNDARY_DE
    base_prompt = boundary + "\n\n" + prompt_template
    if safety_constraint:
        response_agent.system_prompt = base_prompt + "\n\n" + safety_constraint
    else:
        response_agent.system_prompt = base_prompt

    full_response = ""
    final_usage = {}
    async for chunk, chunk_usage, _ in response_agent.generate_stream(full_prompt):
        if chunk:
            full_response += chunk
        if chunk_usage:
            final_usage = chunk_usage
        yield chunk, chunk_usage
    
    # Update observation with final output and usage
    langfuse_context.update_current_observation(
        output=full_response,
        usage=final_usage
    )
