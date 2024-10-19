from Facial_Detection.facial_detection import run_facial_detection
from OpenAI.gpt_gui import start_gpt_gui
import asyncio
import os
# Set environment variables to allow duplicate OpenMP libraries and suppress Hugging Face warnings
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


async def main():
    try:
        await asyncio.gather(
            start_gpt_gui(),  # Run the GPT GUI
            # Run facial detection in a background thread
            asyncio.to_thread(run_facial_detection)
        )
    except Exception as e:
        print("One or more main threads stopped, closing program...")
        print(e)
        return

if __name__ == "__main__":
    asyncio.run(main())
