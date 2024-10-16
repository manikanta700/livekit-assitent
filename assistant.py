import pickle

from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, cli, llm,JobRequest
from livekit.agents.voice_assistant import VoiceAssistant
from livekit.plugins import deepgram, openai, silero,azure
from dotenv import load_dotenv
import psutil 
load_dotenv()
# annoy_index = rag.annoy.AnnoyIndex.load("vdb_data")  # see build_data.py

embeddings_dimension = 1536



async def entrypoint(ctx: JobContext):
    try:

        await ctx.connect()
        print(list(ctx.room.remote_participants.values ()))
        caseid= str(list(ctx.room.remote_participants.values())[0].metadata)
        
    except Exception as e:
        print(e) 
        caseid = """ Based on introduction frame the questions based on his skills on background """

    print(f"Case ID: {caseid}")
    initial_ctx = llm.ChatContext().append(
        role="system",
        text=(
            f"""IDENTITY:
Your name is Miss Minutes, and you are an interviewer conducting a professional interview for a specific role. Your task is to  ask relevant, challenging questions .Your key strength lies in simulating practical, conversational questions that reflect both depth of knowledge and real-world experience .
INSTRUCTIONS:
- Start the interview by greeting the candidate and introducing yourself as Miss Minutes.
-do not respond with list or ask multiple questions at once
- You should STRICTLY evaluate each answer the candidate provides and then respond accordingly with follow-up questions
- DON'T Apperciate the candidate's answers and DON'T provide feedback
-Ask me the questions and wait for my answers. Do not write explanations. Ask me the questions one by one like an interviewer does and wait for my answers.
- *No Numbering or Section Labels:* Do not announce or refer to specific sections or question categories. The conversation should flow naturally.
- If the candidate deviates from the topic, gently guide them back to the relevant focus area.
-Interview Details:
**ASK SELF-INTRODUCTION BY STARTING OF THE INTERVIEW
 { caseid}

"""
            "Your interface with users will be voice. "
            "You should use short and concise responses, and avoiding usage of unpronouncable punctuation."
            "Use the provided context to answer the user's question if needed."
        ),
    )
    

    assistant = VoiceAssistant(
        chat_ctx=initial_ctx,
        vad=silero.VAD.load(),
        stt=deepgram.STT(model="nova-2-general",language="en-IN"),
        llm=openai.LLM.with_ollama(base_url='https://Meta-Llama-3-1-8B-Instruct-aemus.eastus.models.ai.azure.com/v1/'),
        tts=azure.TTS(),
        # min_endpointing_delay=1.5,   
    )

    assistant.start(ctx.room)

    await assistant.say("Hi there ", allow_interruptions=True)



if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
