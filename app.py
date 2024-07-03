import os
import gradio as gr
import boto3
import json
import logging

# Set up logging
logging.basicConfig(filename='/var/log/gradio_app.log', level=logging.DEBUG, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Set up Amazon Bedrock client
bedrock = boto3.client('bedrock-runtime', region_name='us-east-1')

# Function to generate response using Amazon Bedrock
def generate_response(prompt, conversation_history):
    messages = [{"role": "human", "content": msg} for msg in conversation_history.split('\n') if msg.startswith("Human: ")]
    messages.append({"role": "human", "content": prompt})
    
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 300,
        "messages": messages,
        "temperature": 0.7,
        "top_p": 0.95,
    })
    
    try:
        response = bedrock.invoke_model(
            body=body,
            modelId='anthropic.claude-3-sonnet-20240229-v1:0',
            accept='application/json',
            contentType='application/json'
        )
        
        response_body = json.loads(response['body'].read())
        return response_body['content'][0]['text']
    
    except Exception as e:
        logging.error(f"Error invoking Bedrock model: {str(e)}")
        raise

# Chatbot function
def chatbot(message, history):
    try:
        conversation_history = "\n".join([f"Human: {h[0]}\nAssistant: {h[1]}" for h in history])
        logging.info(f"Processing message: {message}")
        response = generate_response(message, conversation_history)
        logging.info(f"Generated response: {response}")
        return response
    except Exception as e:
        logging.error(f"Error in chatbot function: {str(e)}")
        raise


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    
    # Create Gradio interface
    iface = gr.ChatInterface(
        fn=chatbot,
        title="Amazon Bedrock Chatbot",
        description="Chat with an AI powered by Amazon Bedrock",  
        examples=[
            ["What is artificial intelligence?"],
            ["Tell me a joke about programming."],
            ["Explain quantum computing in simple terms."]
        ],
        theme="default"
    )
    
    # Launch the interface
    logging.info(f"Launching Gradio interface on port {port}")
    iface.launch(server_name="0.0.0.0", server_port=port)