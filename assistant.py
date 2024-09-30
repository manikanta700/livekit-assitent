import pickle

from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, cli, llm,JobRequest
from livekit.agents.voice_assistant import VoiceAssistant
from livekit.plugins import deepgram, openai, silero
from dotenv import load_dotenv
import psutil 
load_dotenv()
# annoy_index = rag.annoy.AnnoyIndex.load("vdb_data")  # see build_data.py

embeddings_dimension = 1536



async def entrypoint(ctx: JobContext):
    await ctx.connect()
    caseid= str(list(ctx.room.remote_participants.values ()) [0].metadata)
    # input_list = caseid.split(",")
    # # print(f"input_list----------------{input_list[0]}")
    # async def _enrich_with_rag(assistant: VoiceAssistant, chat_ctx: llm.ChatContext):
    #     # locate the last user message and use it to query the RAG model
    #     # to get the most relevant paragraph
    #     # then provide that as additional context to the LLM
    #     # with open("java.pkl", "rb") as f:
    #     #     java_by_uuid = pickle.load(f)
    #     # with open("Dbms.pkl", "rb") as f:
    #     #     dbms_by_uuid = pickle.load(f)
    #     # with open("python.pkl", "rb") as f:
    #     #     python_by_uuid = pickle.load(f)
    #     # print(f"chat_ctx.messages----------------{chat_ctx.messages}")
    #     # user_msg = chat_ctx.messages[-1]
    #     # print(f"user_msg----------------{user_msg.content}")
    #     # user_embedding = await openai.create_embeddings(
    #     #     input=[user_msg.content],
    #     #     model="text-embedding-3-small",
    #     #     dimensions=embeddings_dimension,
    #     # )

    #     # result = annoy_index.query(user_embedding[0].embedding, n=1)[0]
    #     # print(f"result----------------{result.userdata}")
    #     # try:
    #     #     python = python_by_uuid[result.userdata]
    #     # except:
    #     #     python = " "
    #     # try:
    #     #     java = java_by_uuid[result.userdata]
    #     # except:
    #     #     java = " "
    #     # try:
    #     #     dbms = dbms_by_uuid[result.userdata]
    #     # except:

    #     #     dbms = " "
    #     with open("Technical.pkl", "rb") as f:
    #         Technical_by_uuid = pickle.load(f)
    #     # user_msg=""
    #     # for i in range(len(chat_ctx.messages)):
    #     #     user_msg += chat_ctx.messages[i].content
    #     user_msg=chat_ctx.messages[-1]
    #     # print(f"user_msg----------------{chat_ctx.messages}")
    #     # print(f"user_msg----------------{user_msg.content}")
    #     user_embedding = await openai.create_embeddings(
    #         input=[user_msg.content],
    #         model="text-embedding-3-small",
    #         dimensions=embeddings_dimension,
    #     )
    #     # result = annoy_index.query(user_embedding[0].embedding, n=1)[0]
    #     # print(f"result----------------{result.userdata}")
    #     # Technical= Technical_by_uuid[result.userdata]




    #     chat_ctx.messages[-1].content = (
    #         "Context:\n" + Technical + "\n\nUser Question: " + chat_ctx.messages[-1].content
    #     )

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
        stt=deepgram.STT(),
        llm=openai.LLM(model="gpt-4o"),
        tts=openai.TTS(),
        min_endpointing_delay=1.5,   
    )

    assistant.start(ctx.room)

    await assistant.say("Hi there ", allow_interruptions=True)



if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
