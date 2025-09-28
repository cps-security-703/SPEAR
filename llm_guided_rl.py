import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import random
from collections import defaultdict, deque
import re

class TextAdventureEnvironment:
    """Simple text-based environment where agent navigates and collects items"""
    
    def __init__(self):
        self.locations = {
            'forest': {'items': ['mushroom'], 'connections': ['village', 'cave']},
            'village': {'items': ['bread'], 'connections': ['forest', 'castle']},
            'castle': {'items': ['treasure'], 'connections': ['village']},
            'cave': {'items': ['gem'], 'connections': ['forest']}
        }
        
        self.reset()
    
    def reset(self):
        self.current_location = 'forest'
        self.inventory = []
        self.available_items = {loc: items.copy() for loc, data in self.locations.items() 
                              for items in [data['items']]}
        self.steps = 0
        self.max_steps = 20
        return self.get_state()
    
    def get_state(self):
        state = {
            'location': self.current_location,
            'inventory': self.inventory.copy(),
            'available_items': self.available_items[self.current_location].copy(),
            'connections': self.locations[self.current_location]['connections'].copy(),
            'steps': self.steps
        }
        return state
    
    def get_state_text(self):
        """Convert state to natural language description"""
        state = self.get_state()
        text = f"You are in the {state['location']}. "
        
        if state['available_items']:
            text += f"You see: {', '.join(state['available_items'])}. "
        
        text += f"You can go to: {', '.join(state['connections'])}. "
        
        if state['inventory']:
            text += f"You carry: {', '.join(state['inventory'])}."
        
        return text
    
    def step(self, action):
        """Execute action and return new state, reward, done"""
        reward = 0
        done = False
        
        if action.startswith('go_'):
            location = action[3:]
            if location in self.locations[self.current_location]['connections']:
                self.current_location = location
                reward = 1  # Small reward for exploration
            else:
                reward = -5  # Penalty for invalid move
        
        elif action.startswith('take_'):
            item = action[5:]
            if item in self.available_items[self.current_location]:
                self.available_items[self.current_location].remove(item)
                self.inventory.append(item)
                reward = 10 if item == 'treasure' else 5  # Higher reward for treasure
            else:
                reward = -2  # Penalty for trying to take non-existent item
        
        else:
            reward = -1  # Penalty for invalid action
        
        self.steps += 1
        
        # Check win condition (collected treasure)
        if 'treasure' in self.inventory:
            reward += 50
            done = True
        
        # Check if max steps reached
        if self.steps >= self.max_steps:
            done = True
        
        return self.get_state(), reward, done
    
    def get_valid_actions(self, state):
        """Get list of valid actions for current state"""
        actions = []
        
        # Movement actions
        for location in state['connections']:
            actions.append(f'go_{location}')
        
        # Take item actions
        for item in state['available_items']:
            actions.append(f'take_{item}')
        
        return actions

class LLMAdvisor:
    """LLM that provides advice and context to the RL agent"""
    
    def __init__(self, model_name="distilgpt2"):
        print(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Set pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
    
    def get_advice(self, state_text, available_actions):
        """Generate advice for the current situation"""
        prompt = f"""In a text adventure game:
{state_text}
Available actions: {', '.join(available_actions)}

The best strategy would be to"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=200)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        advice = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        advice = advice[len(prompt):].strip()
        
        return advice
    
    def evaluate_state(self, state_text):
        """Evaluate how good the current state is"""
        prompt = f"""In a text adventure game:
{state_text}

This situation is"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=150)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=30,
                temperature=0.5,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        evaluation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        evaluation = evaluation[len(prompt):].strip()
        
        # Simple scoring based on keywords
        score = 0
        positive_words = ['good', 'great', 'excellent', 'treasure', 'valuable', 'promising']
        negative_words = ['bad', 'poor', 'dangerous', 'empty', 'useless']
        
        for word in positive_words:
            if word in evaluation.lower():
                score += 1
        for word in negative_words:
            if word in evaluation.lower():
                score -= 1
        
        return score

class QLearningAgent:
    """Q-Learning agent that collaborates with LLM"""
    
    def __init__(self, learning_rate=0.1, discount_factor=0.9, epsilon=1.0, epsilon_decay=0.995):
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = 0.01
    
    def state_to_key(self, state):
        """Convert state dict to hashable key"""
        return (
            state['location'],
            tuple(sorted(state['inventory'])),
            tuple(sorted(state['available_items']))
        )
    
    def choose_action(self, state, valid_actions, llm_advisor=None, use_llm_guidance=True):
        """Choose action using epsilon-greedy with optional LLM guidance"""
        state_key = self.state_to_key(state)
        
        if np.random.random() < self.epsilon:
            # Exploration: use LLM guidance or random
            if use_llm_guidance and llm_advisor and np.random.random() < 0.7:
                try:
                    state_text = env.get_state_text() if 'env' in globals() else str(state)
                    advice = llm_advisor.get_advice(state_text, valid_actions)
                    
                    # Try to extract action from advice
                    for action in valid_actions:
                        action_word = action.split('_')[1] if '_' in action else action
                        if action_word.lower() in advice.lower():
                            print(f"LLM suggests: {action} (from advice: '{advice[:50]}...')")
                            return action
                except Exception as e:
                    print(f"LLM guidance failed: {e}")
            
            # Random action as fallback
            return random.choice(valid_actions)
        else:
            # Exploitation: choose best known action
            best_action = max(valid_actions, key=lambda a: self.q_table[state_key][a])
            return best_action
    
    def update_q_value(self, state, action, reward, next_state, valid_next_actions):
        """Update Q-value using Q-learning formula"""
        state_key = self.state_to_key(state)
        next_state_key = self.state_to_key(next_state)
        
        # Find maximum Q-value for next state
        if valid_next_actions:
            max_next_q = max(self.q_table[next_state_key][a] for a in valid_next_actions)
        else:
            max_next_q = 0
        
        # Q-learning update
        old_q = self.q_table[state_key][action]
        new_q = old_q + self.learning_rate * (reward + self.discount_factor * max_next_q - old_q)
        self.q_table[state_key][action] = new_q
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def train_agent(episodes=100, use_llm=True):
    """Train the RL agent with optional LLM collaboration"""
    env = TextAdventureEnvironment()
    agent = QLearningAgent()
    
    # Initialize LLM advisor
    llm_advisor = None
    if use_llm:
        try:
            llm_advisor = LLMAdvisor()
            print("LLM advisor loaded successfully!")
        except Exception as e:
            print(f"Failed to load LLM: {e}")
            use_llm = False
    
    episode_rewards = []
    episode_steps = []
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        
        print(f"\n=== Episode {episode + 1} ===")
        print(env.get_state_text())
        
        while True:
            valid_actions = env.get_valid_actions(state)
            
            if not valid_actions:
                break
            
            # Agent chooses action (with LLM guidance if available)
            action = agent.choose_action(state, valid_actions, llm_advisor, use_llm)
            
            # Execute action
            next_state, reward, done = env.step(action)
            
            # Optional: Get LLM evaluation for reward shaping
            if use_llm and llm_advisor and steps % 3 == 0:  # Every 3 steps
                try:
                    state_text = env.get_state_text()
                    llm_score = llm_advisor.evaluate_state(state_text)
                    reward += llm_score  # Add LLM's evaluation to reward
                    if llm_score != 0:
                        print(f"LLM evaluation bonus: {llm_score}")
                except Exception as e:
                    pass  # Ignore LLM errors
            
            # Update Q-values
            next_valid_actions = env.get_valid_actions(next_state) if not done else []
            agent.update_q_value(state, action, reward, next_state, next_valid_actions)
            
            print(f"Action: {action}, Reward: {reward}, Total: {total_reward + reward}")
            print(env.get_state_text())
            
            total_reward += reward
            steps += 1
            state = next_state
            
            if done:
                break
        
        episode_rewards.append(total_reward)
        episode_steps.append(steps)
        
        print(f"Episode {episode + 1} completed: Reward = {total_reward}, Steps = {steps}")
        print(f"Epsilon: {agent.epsilon:.3f}")
        
        # Show progress every 20 episodes
        if (episode + 1) % 20 == 0:
            avg_reward = np.mean(episode_rewards[-20:])
            avg_steps = np.mean(episode_steps[-20:])
            print(f"\nLast 20 episodes average: Reward = {avg_reward:.2f}, Steps = {avg_steps:.2f}")
    
    return agent, episode_rewards, episode_steps

def test_trained_agent(agent, episodes=5):
    """Test the trained agent"""
    env = TextAdventureEnvironment()
    
    print("\n" + "="*50)
    print("TESTING TRAINED AGENT")
    print("="*50)
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        
        print(f"\n--- Test Episode {episode + 1} ---")
        print(env.get_state_text())
        
        while True:
            valid_actions = env.get_valid_actions(state)
            if not valid_actions:
                break
            
            # Use trained agent (no exploration)
            agent.epsilon = 0  # No exploration during testing
            action = agent.choose_action(state, valid_actions, use_llm_guidance=False)
            
            state, reward, done = env.step(action)
            total_reward += reward
            steps += 1
            
            print(f"Action: {action}, Reward: {reward}")
            print(env.get_state_text())
            
            if done:
                break
        
        print(f"Test episode {episode + 1}: Reward = {total_reward}, Steps = {steps}")
        if 'treasure' in env.inventory:
            print("SUCCESS: Treasure collected!")
        else:
            print("Goal not achieved.")

# Main execution
if __name__ == "__main__":
    print("Starting RL + LLM Collaboration Training...")
    print("This example shows how an LLM can guide and evaluate an RL agent's decisions.")
    
    # Train with LLM collaboration
    print("\n1. Training with LLM guidance...")
    trained_agent, rewards, steps = train_agent(episodes=50, use_llm=True)
    
    # Test the trained agent
    print("\n2. Testing trained agent...")
    test_trained_agent(trained_agent, episodes=3)
    
    # Show training statistics
    print(f"\nTraining completed!")
    print(f"Average reward (last 10 episodes): {np.mean(rewards[-10:]):.2f}")
    print(f"Success rate (treasure collected): {sum(1 for r in rewards[-10:] if r > 45) / 10 * 100:.1f}%")