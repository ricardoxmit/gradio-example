import os
import gradio as gr
import boto3
import json

# Set up Amazon Bedrock client
bedrock = boto3.client('bedrock-runtime', region_name='us-east-1')

# Function to generate response using Amazon Bedrock
def generate_response(prompt, conversation_history):
    # Prepare the request body
    body = json.dumps({
        "prompt": f"{conversation_history}\nHuman: {prompt}\nAssistant:",
        "max_tokens_to_sample": 300,
        "temperature": 0.7,
        "top_p": 0.95,
    })
    
    # Make a request to Amazon Bedrock
    response = bedrock.invoke_model(
        body=body,
        modelId='anthropic.claude-3-sonnet-20240229-v1:0',  # Claude Sonnet model
        accept='application/json',
        contentType='application/json'
    )
    
    # Parse the response
    response_body = json.loads(response['body'].read())
    return response_body['completion'].strip()

# Chatbot function
def chatbot(message, history):
    conversation_history = "\n".join([f"Human: {h[0]}\nAssistant: {h[1]}" for h in history])
    response = generate_response(message, conversation_history)
    return response

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    
    # Create Gradio interface
    iface = gr.ChatInterface(
        fn=chatbot,
        title="Amazon Bedrock Chatbot",
        description="Chat with an AI powered by Amazon Bedrock",  # Removed extra 'b'
        examples=[
            ["What is artificial intelligence?"],
            ["Tell me a joke about programming."],
            ["Explain quantum computing in simple terms."]
        ],
        theme="default"
    )
    
    # Launch the interface
    iface.launch(server_name="0.0.0.0", server_port=port)