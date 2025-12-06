import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login

class LlamaAgent:
    def __init__(self, model_name="meta-llama/Llama-3.1-8B", token_file="keys.txt"):
        """
        Initialize the Llama agent with authentication and model loading.
        
        Args:
            model_name (str): Hugging Face model identifier
            token_file (str): File containing the Hugging Face token
        """
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load and authenticate with Hugging Face token
        self._authenticate(token_file)
        
        # Load model and tokenizer
        self._load_model()
    
    def _authenticate(self, token_file):
        """Authenticate with Hugging Face using token from file."""
        try:
            with open(token_file, 'r') as f:
                token = f.read().strip()
            login(token=token)
            print("✅ Successfully authenticated with Hugging Face")
        except Exception as e:
            print(f"❌ Authentication failed: {e}")
            raise
    
    def _load_model(self):
        """Load the tokenizer and model."""
        try:
            print(f"🔄 Loading {self.model_name}...")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with optimizations
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                low_cpu_mem_usage=True
            )
            
            if not torch.cuda.is_available():
                self.model = self.model.to(self.device)
            
            print(f"✅ Model loaded successfully on {self.device}")
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            raise
    
    def generate_response(self, user_input, max_length=512, temperature=0.7, do_sample=True):
        """
        Generate a response to user input using the Llama model.
        
        Args:
            user_input (str): The user's question or prompt
            max_length (int): Maximum length of generated response
            temperature (float): Sampling temperature (higher = more creative)
            do_sample (bool): Whether to use sampling or greedy decoding
            
        Returns:
            str: Generated response from the model
        """
        try:
            # Format the prompt
            prompt = f"User: {user_input}\nAssistant: "
            
            # Tokenize input
            inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=len(inputs[0]) + max_length,
                    temperature=temperature,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.eos_token_id,
                    attention_mask=torch.ones_like(inputs)
                )
            
            # Decode and extract response
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the assistant's response
            response = full_response.split("Assistant: ")[-1].strip()
            
            return response
            
        except Exception as e:
            return f"❌ Error generating response: {e}"
    
    def chat_loop(self):
        """Interactive chat loop with the user."""
        print("🤖 Llama Agent initialized! Type 'quit' or 'exit' to stop.")
        print("=" * 50)
        
        while True:
            try:
                # Get user input
                user_input = input("\n👤 You: ").strip()
                
                # Check for exit commands
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("👋 Goodbye!")
                    break
                
                if not user_input:
                    print("Please enter a question or message.")
                    continue
                
                # Generate and display response
                print("\n🤖 Assistant: ", end="")
                response = self.generate_response(user_input)
                print(response)
                
            except KeyboardInterrupt:
                print("\n👋 Chat interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")

def main():
    """Main function to run the LLM agent."""
    try:
        # Initialize the agent
        agent = LlamaAgent()
        
        # Start interactive chat
        agent.chat_loop()
        
    except Exception as e:
        print(f"❌ Failed to initialize agent: {e}")
        print("Make sure you have the required dependencies installed:")
        print("pip install -r requirements.txt")

if __name__ == "__main__":
    main()
