import gradio as gr
from huggingface_hub import InferenceClient
from transformers import pipeline

"""
For more information on `huggingface_hub` Inference API support, please check the docs: https://huggingface.co/docs/huggingface_hub/v0.22.2/en/guides/inference
"""
print("Starting......")


chatbot = pipeline("text-generation", model="gpt2")

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
    messages = [{"role": "system", "content": system_message}]
    messages.append({"role": "user", "content": message})
    print("getting response", messages )
    response = chatbot(messages) 
    print("got response", response )
    return response[-1]['generated text'][-1]['content']
   


"""
For information on how to customize the ChatInterface, peruse the gradio docs: https://www.gradio.app/docs/chatinterface
"""
demo = gr.ChatInterface(
    respond,
    additional_inputs=[
        gr.Textbox(value="You are a friendly Chatbot.", label="System message"),
        gr.Slider(minimum=1, maximum=2048, value=512, step=1, label="Max new tokens"),
        
        
        ),
    ],
)


if __name__ == "__main__":
    demo.launch()
