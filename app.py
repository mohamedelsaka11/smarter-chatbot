import gradio as gr # Create interactive interfaces (GUI)
from huggingface_hub import InferenceClient # tool to connect to the cloud 
from transformers import pipeline #A library with pre-trained models

"""
For more information on `huggingface_hub` Inference API support, please check the docs: https://huggingface.co/docs/huggingface_hub/v0.22.2/en/guides/inference
"""
print("Starting......")

# loading the Microsoft model
chatbot = pipeline("text-generation", model="microsoft/Phi-3-mini-4k-instruct")

print("defining function")

def respond(
    message,
    history: list[tuple[str, str]],
    system_message,
    max_tokens,
    temperature,
    top_p,
):
    print("enter response")
    conversation = f"{system_message}\n"
    for user_msg, bot_msg in history:
        conversation += f"User: {user_msg}\nAssistant: {bot_msg}\n"
    conversation += f"User: {message}\nAssistant:"
    print("getting response", conversation)

    response = chatbot(
        conversation,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p
    )

    print("got response", response)
    return response[0]['generated_text'].split("Assistant:")[-1].strip()
   


"""
For information on how to customize the ChatInterface, peruse the gradio docs: https://www.gradio.app/docs/chatinterface
"""
demo = gr.ChatInterface(
    respond,
    additional_inputs=[
        gr.Textbox(value="You are a friendly Chatbot.", label="System message"),
        gr.Slider(minimum=1, maximum=2048, value=512, step=1, label="Max new tokens"),
        
        
    
    ],
)


if __name__ == "__main__":
    demo.launch()
