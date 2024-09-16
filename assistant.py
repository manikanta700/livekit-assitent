import asyncio
from typing import Annotated

from livekit import agents, rtc
from livekit.agents import JobContext, WorkerOptions, cli, tokenize, tts
from livekit.agents.llm import (
    ChatContext,
    ChatImage,
    ChatMessage,
)
from livekit.agents.voice_assistant import VoiceAssistant
from livekit.plugins import deepgram, openai, silero


from dotenv import load_dotenv
load_dotenv()

class AssistantFunction(agents.llm.FunctionContext):
    """This class is used to define functions that will be called by the assistant."""

    @agents.llm.ai_callable(
        description=(
            "Called when asked to evaluate something that would require vision capabilities,"
            "for example, an image, video, or the webcam feed."
        )
    )
    async def image(
        self,
        user_msg: Annotated[
            str,
            agents.llm.TypeInfo(
                description="The user message that triggered this function"
            ),
        ],
    ):
        print(f"Message triggering vision capabilities: {user_msg}")
        return None


async def get_video_track(room: rtc.Room):
    """Get the first video track from the room. We'll use this track to process images."""

    video_track = asyncio.Future[rtc.RemoteVideoTrack]()

    for _, participant in room.remote_participants.items():
        for _, track_publication in participant.track_publications.items():
            if track_publication.track is not None and isinstance(
                track_publication.track, rtc.RemoteVideoTrack
            ):
                video_track.set_result(track_publication.track)
                print(f"Using video track {track_publication.track.sid}")
                break

    return await video_track

import asyncio
import psutil
import os
from typing import Annotated

from livekit import agents, rtc
from livekit.agents import JobContext, WorkerOptions, cli, tokenize, tts
from livekit.agents.llm import (
    ChatContext,
    ChatImage,
    ChatMessage,
)
from livekit.agents.voice_assistant import VoiceAssistant
from livekit.plugins import deepgram, openai, silero

from dotenv import load_dotenv
load_dotenv()

class AssistantFunction(agents.llm.FunctionContext):
    """This class is used to define functions that will be called by the assistant."""

    @agents.llm.ai_callable(
        description=(
            "Called when asked to evaluate something that would require vision capabilities,"
            "for example, an image, video, or the webcam feed."
        )
    )
    async def image(
        self,
        user_msg: Annotated[
            str,
            agents.llm.TypeInfo(
                description="The user message that triggered this function"
            ),
        ],
    ):
        print(f"Message triggering vision capabilities: {user_msg}")
        return None


async def get_video_track(room: rtc.Room):
    """Get the first video track from the room. We'll use this track to process images."""

    video_track = asyncio.Future[rtc.RemoteVideoTrack]()

    for _, participant in room.remote_participants.items():
        for _, track_publication in participant.track_publications.items():
            if track_publication.track is not None and isinstance(
                track_publication.track, rtc.RemoteVideoTrack
            ):
                video_track.set_result(track_publication.track)
                print(f"Using video track {track_publication.track.sid}")
                break

    return await video_track


# Function to print current memory usage
def print_memory_usage():
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info().rss / (1024 ** 2)  # in MB
    print(f"Current memory usage: {memory_info:.2f} MB")


async def entrypoint(ctx: JobContext):
    await ctx.connect()
    print(f"Room name: {ctx.room.name}")
    print_memory_usage()  # Check RAM usage after connection

    chat_context = ChatContext(
        messages=[
            ChatMessage(
                role="system",
                content=(
                    """


Your name is Alloy, and you are an interviewer conducting a professional interview for a specific role. Your task is to assess the candidate’s qualifications and skills by asking relevant, challenging questions based on their resume and the latest interview patterns for the company.

- You will receive the candidate's resume and interview details (including company name, role, and job description) in the first message of the conversation. This message is automated; do not request the resume or any other details from the candidate.
- Immediately after receiving the resume and job details, fetch the latest interview patterns, questions, and expectations for the specific company and role from available internet sources.
- Use these up-to-date resources to tailor your questions according to the company’s expectations, ensuring that the interview is aligned with current practices (e.g., starting with a self-introduction in companies like TCS).

### Interview Guidelines:
- **Duration:** The interview should last at least 30 minutes in most cases. The total duration must strictly adhere to the given time limit.
- **Question Flow:** 
  - Ask one question at a time, wait for the candidate to answer, and only then proceed to the next question.
  - Analyze the candidate's answers carefully, and if the response is insufficient or unclear, ask follow-up questions to delve deeper until the response is satisfactory.
- **First Phase:** Focus on domain-related questions relevant to the candidate's role, expertise, and the company’s expectations.
- **Final 10 Minutes:** In the last 10 minutes, gradually transition to broader topics such as:
  - Qualifications and experience
  - Interest in the role and organization
  - Creativity and innovation
  - Team-building and interpersonal skills
  - Knowledge of the company's mission, values, and goals

- **No Numbering or Section Labels:** Do not announce or refer to specific sections or question categories. The conversation should flow naturally.
- Ensure a variety of questions without excessive repetition.
- If the candidate deviates from the topic, gently guide them back to the relevant focus area.
- At the end of the interview, assess the candidate’s overall performance without revealing the evaluation process to the candidate.
- Maintain a professional and neutral tone throughout.

### TCS-Specific Interview Pattern:
If the job description, company, or any detail indicates that the interview is for a TCS Prime/Digital/Ninja role, the interview must follow the TCS-specific pattern:

1. **Technical Interview (75% of the total interview duration):**
   - Focus primarily on technical skills and projects.
   - Ask at least 20 questions related to the skills mentioned in the resume.
   - Ask at least 10 questions about the candidate's projects.
   - The level of difficulty should vary based on the role:
     - **Prime:** Highest level of difficulty
     - **Digital:** Intermediate level of difficulty
     - **Ninja:** Beginner to intermediate level of difficulty
   - Analyze the candidate’s answers, and if the response is inadequate or unclear, ask follow-up questions to delve deeper into the topic.
   - Make it clear to the candidate that this is the **Technical Round**.
   - Inform the candidate whether they qualify for the next round at the end of the technical interview.

2. **Managerial Round:**
   - A mix of technical and behavioral questions, assessing problem-solving under pressure, career goals, and motivation for joining TCS.
   - Make it clear to the candidate that this is the **Managerial Round**.
   - Inform the candidate if they qualify for the next round at the end of this round.

3. **HR Round:**
   - Focus on cultural fit, asking questions about work experience, education, career goals, and interests.
   - Make it clear to the candidate that this is the **HR Round**.
   - Inform the candidate if they qualify after this round as well.

When conducting an interview for TCS, ensure that these steps are followed, and clearly communicate the round names and results after each round to the candidate.


                        """
                ),
            )
        ]
    )

    gpt = openai.LLM(model="gpt-4o")

    openai_tts = tts.StreamAdapter(
        tts=openai.TTS(voice="alloy"),
        sentence_tokenizer=tokenize.basic.SentenceTokenizer(),
    )

    latest_image: rtc.VideoFrame | None = None

    assistant = VoiceAssistant(
        vad=silero.VAD.load(),  # We'll use Silero's Voice Activity Detector (VAD)
        stt=deepgram.STT(),  # We'll use Deepgram's Speech To Text (STT)
        llm=gpt,
        tts=openai_tts,  # We'll use OpenAI's Text To Speech (TTS)
        fnc_ctx=AssistantFunction(),
        chat_ctx=chat_context,
    )

    chat = rtc.ChatManager(ctx.room)

    # Check memory after assistant starts
    assistant.start(ctx.room)
    print_memory_usage()  # Check RAM usage after starting assistant

    async def _answer(text: str, use_image: bool = False):
        """
        Answer the user's message with the given text and optionally the latest
        image captured from the video track.
        """
        content: list[str | ChatImage] = [text]
        if use_image and latest_image:
            content.append(ChatImage(image=latest_image))

        chat_context.messages.append(ChatMessage(role="user", content=content))

        stream = gpt.chat(chat_ctx=chat_context)
        await assistant.say(stream, allow_interruptions=True)

    @chat.on("message_received")
    def on_message_received(msg: rtc.ChatMessage):
        """This event triggers whenever we get a new message from the user."""
        if msg.message:
            asyncio.create_task(_answer(msg.message, use_image=False))

    @assistant.on("function_calls_finished")
    def on_function_calls_finished(called_functions: list[agents.llm.CalledFunction]):
        """This event triggers when an assistant's function call completes."""
        if len(called_functions) == 0:
            return

        user_msg = called_functions[0].call_info.arguments.get("user_msg")
        if user_msg:
            asyncio.create_task(_answer(user_msg, use_image=True))

    await asyncio.sleep(1)
    await assistant.say("Hi there! ?", allow_interruptions=True)

    # Check memory while processing video frames
    while ctx.room.connection_state == rtc.ConnectionState.CONN_CONNECTED:
        video_track = await get_video_track(ctx.room)

        async for event in rtc.VideoStream(video_track):
            latest_image = event.frame
            print_memory_usage()  # Check RAM usage after processing each frame


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
