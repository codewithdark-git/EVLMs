import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from typing import Dict, Optional, List, Union, Any

class MedicalLanguageDecoder(nn.Module):
    """Language decoder for medical report generation"""
    
    def __init__(self,
                 model_name: str = 'google/gemma-3-1b-pt',
                 vocab_size: int = 262144,  # Gemma-3B vocab size
                 max_length: int = 512,
                 lora_r: int = 16,
                 lora_alpha: int = 32,
                 lora_dropout: float = 0.1):
        """
        Args:
            model_name: Name of the base language model
            vocab_size: Vocabulary size
            max_length: Maximum sequence length
            lora_r: LoRA rank
            lora_alpha: LoRA alpha
            lora_dropout: LoRA dropout
        """
        super().__init__()
        
        # Load tokenizer for Gemma
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load Gemma language model with eager attention (recommended)
        self.language_model = AutoModelForCausalLM.from_pretrained(
            model_name
        )

        # Unfreeze last transformer block and LM head for better adaptation
        for name, param in self.language_model.named_parameters():
            if 'lm_head' in name:  # For Gemma-3B, last block
                param.requires_grad = True

        # Vision-to-text projection (match Gemma hidden size)
        self.vision_projection = nn.Sequential(
            nn.Linear(768, 1024),
            nn.GELU(),
            nn.LayerNorm(1024),
            nn.Linear(1024, self.language_model.config.hidden_size)
        )

        # Medical knowledge embeddings
        self.medical_embeddings = self._init_medical_embeddings(vocab_size)

        # LoRA for efficient fine-tuning (Gemma compatible)
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Gemma uses these
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )
        self.language_model = get_peft_model(self.language_model, lora_config)
    
    def _init_medical_embeddings(self, vocab_size: int = 10000) -> nn.Embedding:
        """Initialize medical concept embeddings
        
        Args:
            vocab_size: Size of medical vocabulary
        
        Returns:
            Medical concept embeddings
        """
        return nn.Embedding(vocab_size, 768)
    
    def forward(self,
                visual_features: torch.Tensor,
                text_input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            visual_features: Visual features from encoder (B, N, D)
            text_input_ids: Input text token IDs
            attention_mask: Attention mask for text
            labels: Labels for text generation
        
        Returns:
            Dictionary containing model outputs
        """
        batch_size = visual_features.shape[0]
        
        # Project visual features to language space
        visual_tokens = self.vision_projection(visual_features).to(self.language_model.dtype)  # [B, N, n_embd]
        
        if text_input_ids is not None:
            # Training mode - teacher forcing
            text_embeddings = self.language_model.get_input_embeddings()(text_input_ids)
            
            # Concatenate visual and text embeddings
            combined_embeddings = torch.cat([visual_tokens, text_embeddings], dim=1)
            
            # Create attention mask for combined input
            visual_attention = torch.ones(
                batch_size, visual_tokens.shape[1],
                device=visual_tokens.device,
                dtype=attention_mask.dtype
            )
            combined_attention = torch.cat([visual_attention, attention_mask], dim=1)
            
            # Create labels for combined input
            if labels is not None:
                visual_labels = torch.full(
                    (batch_size, visual_tokens.shape[1]),
                    -100,  # Ignore index for loss calculation
                    dtype=torch.long,
                    device=labels.device
                )
                combined_labels = torch.cat([visual_labels, labels], dim=1)
            else:
                combined_labels = None

            # Forward pass through language model
            outputs = self.language_model(
                inputs_embeds=combined_embeddings,
                attention_mask=combined_attention,
                labels=combined_labels,
                output_hidden_states=True
            )
            
            return {
                'loss': outputs.loss,
                'logits': outputs.logits,
                'hidden_states': outputs.hidden_states,
                'attention_mask': combined_attention
            }
        else:
            # Inference mode - autoregressive generation
            return self.generate_explanation(visual_tokens)
    
    @torch.no_grad()
    def generate_explanation(self,
                           visual_features: torch.Tensor,
                           max_length: int = 200,
                           num_beams: int = 4,
                           temperature: float = 0.7,
                           top_p: float = 0.9,
                           prompt: str = "Findings: ") -> Dict[str, Any]:
        """
        Generate medical explanation from visual features
        
        Args:
            visual_features: Visual features in language space
            max_length: Maximum generation length
            num_beams: Number of beams for beam search
            temperature: Sampling temperature
            top_p: Nucleus sampling probability
        
        Returns:
            Dictionary containing generated text and metadata
        """
        # Project visual features to language space
        visual_tokens = self.vision_projection(visual_features).to(self.language_model.dtype)

        # Encode prompt and get embeddings
        prompt_encoding = self.tokenizer(
            [prompt] * visual_features.shape[0],
            return_tensors='pt',
            padding=True
        )
        prompt_input_ids = prompt_encoding['input_ids'].to(visual_tokens.device)
        prompt_embeddings = self.language_model.get_input_embeddings()(prompt_input_ids)

        # Concatenate visual tokens and prompt embeddings
        current_embeddings = torch.cat([visual_tokens, prompt_embeddings], dim=1)

        # Generate text using greedy decoding for more stable output
        outputs = self.language_model.generate(
            inputs_embeds=current_embeddings,
            max_length=max_length,
            num_beams=1,
            do_sample=False,
            early_stopping=True
        )

        # Log generated token IDs for debugging
        print("Generated token IDs:", outputs)
        generated_text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        print("Generated text:", generated_text)

        return {
            'explanations': generated_text,
            'tokens': outputs
        }
    
    def encode_text(self, text: Union[str, List[str]]) -> Dict[str, torch.Tensor]:
        """Encode text using tokenizer
        
        Args:
            text: Input text or list of texts
        
        Returns:
            Dictionary containing encoded text
        """
        if isinstance(text, str):
            text = [text]
        
        encoding = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask']
        } 
